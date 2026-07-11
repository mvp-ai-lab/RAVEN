"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import gc
import logging
import os
import time
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from project.diffusion.samplers import SAMPLER_REGISTRY, BaseSampler
from project.diffusion.schedules import SCHEDULE_REGISTRY, BaseSchedule
from project.diffusion.timesteps import TIMESTEP_REGISTRY, BaseTimesteps
from project.engines import ENGINE_REGISTRY
from project.engines.base_engine import BaseEngine
from project.meta_models import BaseForwardInput
from project.utils import comm, fs, running
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.mfu import disable_flops_accumulate
from project.utils.random import RandomState

logger = logging.getLogger()


@dataclass
class DiffusionFinetuningConfig(Dataclass):
    # diffusion
    timestep: CfgNode = field(default=None)
    schedule: CfgNode = field(default=None)
    sampler: CfgNode = field(default=None)
    # general
    validation: List[CfgNode] = field(default_factory=list)
    save_before_train: bool = field(default=False)
    val_before_train: bool = field(default=True)
    val_backbone: bool = field(default=True)
    val_ema_model: bool = field(default=True)
    step_models: List[str] = field(default_factory=lambda: ["backbone"])
    # training
    training_steps: int = field(default=1000000)
    ga_steps: int = field(default=1)  # gradient accumulation steps
    prepare_before_sync: bool = field(default=True)  # whether to prepare inputs before sync for unified parallel
    prepare_after_sync: bool = field(default=False)  # whether to prepare inputs after sync for unified parallel
    uncond_prob: float = field(default=0.0)
    loss_type: str = field(default="v_lerp")  # x_0, x_T, v_cos, v_lerp, x_t


class DiffusionFinetuning(BaseEngine):
    backbone: nn.Module

    training_timesteps: BaseTimesteps
    schedule: BaseSchedule
    sampler: BaseSampler

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.setup_writer(cfg)

        self.engine_config = DiffusionFinetuningConfig(**self.config["_config"])
        self.backbone = self.models["backbone"]
        self.configure_diffusion()

        self.sync_inputs = lambda inputs: [inputs]

    def configure_diffusion(self):
        # timestep
        timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.timestep["_class_name"])
        self.training_timesteps = timestep_cls(**self.engine_config.timestep["_config"])
        # schedule
        schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.schedule["_class_name"])
        self.schedule = schedule_cls(**self.engine_config.schedule["_config"])
        # sampler
        sampler_cls = SAMPLER_REGISTRY.get(self.engine_config.sampler["_class_name"])
        self.sampler = sampler_cls(schedule=self.schedule, **self.engine_config.sampler["_config"])

    def run(self):
        # save before train
        if self.engine_config.save_before_train:
            logger.info(f"saving before train...")
            self.save(save_before_train=True)

        # val before train
        if self.engine_config.val_before_train:
            logger.info(f"validation before train...")
            self.validate()

        # training progress
        pbar = tqdm(
            initial=self.iter,
            total=self.engine_config.training_steps * self.engine_config.ga_steps,
            dynamic_ncols=True,
            disable=comm.get_local_rank() != 0,
            desc="Training",
        )
        self.start_iter = self.iter

        # Set up manual garbage collection.
        # Manual gc prevents performance slowdown over time.
        gc.disable()
        gc.collect()

        logger.info(f"start training...")
        world_size = comm.get_world_size()
        rank = comm.get_rank()
        num_sp_groups = comm.get_world_size()
        sp_group_id = comm.get_rank()

        train_start_time = time.perf_counter()
        start_time = train_start_time
        while self.iter < self.engine_config.training_steps * self.engine_config.ga_steps:
            batch = next(self.dataloader)
            end_time = time.perf_counter()
            self.average_meter.put_scalar("timer/dataload", end_time - start_time)

            if self.engine_config.prepare_before_sync:
                self.mfu_start.record()
                seed = self.persistence_config.seed + self.iter * world_size + rank
                rng = RandomState(seed=seed)
                batch = self.prepare_before_sync(batch, rng)

                self.mfu_end.record()
                mfu_dict = self.get_mfu(show=self.iter == self.start_iter)
                mfu_dict = {f"{k}/before_sync": v for k, v in mfu_dict.items()}
                self.average_meter.update(mfu_dict)
                self.average_meter.put_scalar("timer/prepare_before_sync", time.perf_counter() - end_time)
                end_time = time.perf_counter()

            for sp_idx, batch in enumerate(self.sync_inputs(batch)):
                seed = self.persistence_config.seed + self.iter * num_sp_groups + sp_group_id
                rng = RandomState(seed=seed)

                if self.engine_config.prepare_after_sync:
                    self.mfu_start.record()
                    batch = self.prepare_after_sync(batch, rng)

                    self.mfu_end.record()
                    mfu_dict = self.get_mfu(show=self.iter == self.start_iter)
                    mfu_dict = {f"{k}/after_sync": v for k, v in mfu_dict.items()}
                    self.average_meter.update(mfu_dict)
                    self.average_meter.put_scalar("timer/prepare_after_sync", time.perf_counter() - end_time)
                    end_time = time.perf_counter()

                # lr schedulers step
                for k, v in self.lr_schedulers.items():
                    v.step(self.iter)

                # training loop including fwd, bwd and optim step
                # return main metrics for progress bar logging, e.g. loss and grad norm
                self.mfu_start.record()
                log_dict = self.training_loop(batch, rng)

                # log mfu for training loop
                self.mfu_end.record()
                perf_dict = self.get_mfu(show=self.iter == self.start_iter)
                perf_dict = {f"{k}/training_loop": v for k, v in perf_dict.items()}

                # update progress bar
                train_time = time.perf_counter() - end_time
                perf_dict["timer/train"] = train_time  # show on progress bar
                batch_time = time.perf_counter() - start_time
                perf_dict["timer/batch"] = batch_time  # show on progress bar

                pbar.set_postfix(perf_dict)
                pbar.update()
                log_dict.update(perf_dict)
                self.average_meter.update(log_dict)

                if (self.iter + 1) % (self.engine_config.ga_steps * self.persistence_config.log_interval) == 0:
                    step = (self.iter + 1) // self.engine_config.ga_steps
                    run_step = (self.iter + 1 - self.start_iter) // self.engine_config.ga_steps
                    timer_per_step = (time.perf_counter() - train_start_time) / run_step

                    # sync between all ranks
                    self.average_meter.sync()                   # mainly logged by engine
                    running.get_running_average_meter().sync()  # logged inside model or anywhere else
                    running.get_running_accumulator().sync()    # logged inside model or anywhere else

                    # log metrics
                    if self.writer is not None:
                        writer_dict = self.average_meter.avg
                        writer_dict.update(running.get_running_average_meter().avg)
                        writer_dict.update(running.get_running_accumulator().sum)
                        writer_dict["timer/step"] = timer_per_step
                        self.writer.log(writer_dict, step=step)

                    # log to stdout and logfile
                    sstring = f"[Training {self.iter+1:07d}/{self.engine_config.training_steps * self.engine_config.ga_steps:07d}]"
                    table_data = [[k, f"{v:.6f}"] for k, v in self.average_meter.items()]
                    table_data.append(["timer/step", f"{timer_per_step:.6f}"])

                    # memory info
                    memory_free, memory_total = torch.cuda.mem_get_info()
                    memory_used = (memory_total - memory_free) / (1024 ** 3)  # in GB
                    num_alloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
                    table_data.append(["memory_used (GB)", f"{memory_used:.2f}"])
                    table_data.append(["num_alloc_retries", f"{num_alloc_retries}"])

                    N_COLS = 2
                    nrows = math.ceil(len(table_data) / N_COLS)
                    table_data += [["", ""]] * (nrows * N_COLS - len(table_data))
                    table_data = [[x for j in range(N_COLS) for x in table_data[j * nrows + i]] for i in range(nrows)]
                    sstring += "\n" + tabulate(
                        table_data,
                        headers=["Metric", "Value"] * N_COLS,
                        tablefmt="heavy_outline",
                        numalign="left",
                        stralign="left",
                    )

                    logger.info(sstring)
                    self.average_meter.reset()
                    running.get_running_average_meter().reset()
                    running.get_running_accumulator().reset()

                    # Manual gc prevents performance slowdown over time.
                    gc.collect()

                if (
                    (self.iter + 1) >= self.engine_config.ga_steps * self.persistence_config.save_start_iter
                    and (self.iter + 1) % (self.engine_config.ga_steps * self.persistence_config.save_interval) == 0
                ):
                    self.save()

                if (
                    (self.iter + 1) >= self.engine_config.ga_steps * self.persistence_config.val_start_iter
                    and (self.iter + 1) % (self.engine_config.ga_steps * self.persistence_config.val_interval) == 0
                ):
                    self.validate()

                self.iter += 1
                start_time = time.perf_counter()

        if self.writer is not None:
            self.writer.finish(exit_code=0)
        comm.barrier()
        logger.info("Training is complete.")

    def optimize(self, step_models: List[str] = ["backbone"]):
        if (self.iter + 1) % self.engine_config.ga_steps != 0:
            return dict()

        norm_dict = dict()
        for name in step_models:
            model = self.models[name]
            optims = [optim for k, optim in self.optimizers.items() if k.startswith(name)]

            # clip grad value
            if self.config.models[name].clip_grad_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), self.config.models[name].clip_grad_value)

            # clip grad norm
            if isinstance(model, FSDP):
                norm = model.clip_grad_norm_(self.config.models[name].clip_grad_norm)
            else:
                assert isinstance(model, FSDPModule), f"model {name} is not FSDP or FSDPModule"
                norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.models[name].clip_grad_norm)

            # check nan or inf
            has_nan = torch.isnan(norm).any()
            has_inf = torch.isinf(norm).any()
            if has_nan or has_inf:
                if has_nan:
                    running.get_running_accumulator().put_scalar(f"running/check_nan/{name}", 1)
                    logger.info(f"grad norm is nan for {name} at iter {self.iter}, skip optim step and zero grad")
                if has_inf:
                    running.get_running_accumulator().put_scalar(f"running/check_inf/{name}", 1)
                    logger.info(f"grad norm is inf for {name} at iter {self.iter}, skip optim step and zero grad")
                for optim in optims:
                    optim.zero_grad()
                continue
            norm_dict[f"grad_norm/{name}"] = norm.item()

            # update grad
            for optim in optims:
                optim.step()
                optim.zero_grad()

            # update ema model
            if self.config.models[name].ema.enabled:
                ema_model = self.models[f"{name}_ema"]
                ema_decay = self.config.models[name].ema.decay
                with torch.no_grad():
                    for src, tgt in zip(model.parameters(), ema_model.parameters()):
                        if not src.requires_grad:
                            continue
                        tgt.detach().mul_(ema_decay).add_(src, alpha=1.-ema_decay)

        return norm_dict

    def training_loop(
        self,
        inputs: Tuple[BaseForwardInput, BaseForwardInput],
        rng: RandomState
    ) -> Dict[str, float]:
        # log data stats
        pos_inputs, neg_inputs = inputs
        running.get_running_accumulator().put_scalar(f"data/num_total_tokens", sum(pos_inputs.seqlens))
        running.get_running_accumulator().put_scalar(f"data/num_samples", pos_inputs.batch_size)
        log_dict = dict()

        # gradient accumulation context
        if isinstance(self.backbone, FSDP) and (self.iter + 1) % self.engine_config.ga_steps != 0:
            ctx = self.backbone.no_sync()
        else:
            ctx = nullcontext()
        if isinstance(self.backbone, FSDPModule):
            self.backbone.set_requires_gradient_sync(requires_gradient_sync=(self.iter + 1) % self.engine_config.ga_steps == 0)

        # forward and backward pass
        with ctx:
            running.set_training_phase(running.TrainingPhase.IN_FORWARD)
            loss = self.training_step(pos_inputs, neg_inputs, rng)
            running.set_training_phase(running.TrainingPhase.IN_BACKWARD)
            log_dict["train/loss"] = loss.item()
            loss = loss / self.engine_config.ga_steps
            loss.backward()

        # optimization step
        running.set_training_phase(running.TrainingPhase.IN_OPTIMIZATION)
        norm_dict = self.optimize(self.engine_config.step_models)

        log_dict.update(norm_dict)
        return log_dict

    def training_step(
        self,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        rng: RandomState
    ) -> torch.Tensor:
        # trigger 1st all-gather earlier
        if isinstance(self.backbone, FSDPModule):
            self.backbone.unshard()

        # drop condition according to uncond_prob
        batch_size = pos_inputs.batch_size
        uncond_mask = torch.rand(batch_size, device=self.device, generator=rng.torch_cuda_generator) < self.engine_config.uncond_prob
        inputs = self.drop_condition(pos_inputs, neg_inputs, uncond_mask, rng)

        # sample timesteps and noises
        ts = self.sample_timesteps(inputs, self.training_timesteps, rng)
        noises = self.sample_noises(inputs, rng)
        inputs = self.set_timesteps(inputs, ts)
        inputs = self.set_noises(inputs, noises)

        # add noises
        noisy_latents = self.add_noises(self.schedule, inputs)
        inputs = self.set_noisy_latents(inputs, noisy_latents)

        # pred
        pred = self.pred(self.backbone, inputs)

        # convert and calculate loss
        pred = self.convert_pred(
            schedule=self.schedule,
            inputs=inputs,
            pred=pred,
            loss_type=self.engine_config.loss_type
        )
        target = self.convert_target(
            schedule=self.schedule,
            inputs=inputs,
            loss_type=self.engine_config.loss_type
        )

        loss = self.loss_fn(inputs, pred, target)
        return loss

    @torch.no_grad()
    @disable_flops_accumulate()  # we found mfu logging outside training loop cause compiled model to be slower
    def validate(self):
        validation_start_time = time.perf_counter()
        running.set_training_phase(running.TrainingPhase.IN_EVAL)  # use None indicate eval
        torch.cuda.empty_cache()

        model_names = []
        if self.engine_config.val_backbone:
            model_names.append("backbone")
        if "backbone_ema" in self.models and self.engine_config.val_ema_model:
            model_names.append("backbone_ema")
        if not model_names:
            logger.info(f"[Validation @ Steps {self.iter+1:07d}] skipped because val_backbone and val_ema_model are both disabled")
            return

        model_training_modes = {model_name: self.models[model_name].training for model_name in model_names}
        for model_name in model_names:
            self.models[model_name].train(False)

        local_dir = os.path.join(self.output_dir, "validation", f"{(self.iter + 1):07d}")
        os.makedirs(local_dir, exist_ok=True)
        save_dir = os.path.join(self.save_dir, "validation", f"{(self.iter + 1):07d}")
        fs.mkdir(save_dir)

        for validation_idx, validation_cfg in enumerate(self.engine_config.validation):
            validation_cls = ENGINE_REGISTRY.get(validation_cfg["_class_name"])
            validation_name = validation_cfg.get("name") or f"idx{validation_idx}"
            for model_name in model_names:
                model = self.models[model_name]
                assert hasattr(validation_cls, "validate"), f"{validation_cls} does not have validate function"
                obj = validation_cls.validate(
                    meta_model=self,
                    model=model,
                    schedule=self.schedule,
                    infer_cfg=validation_cfg["_config"],
                    output_dir=os.path.join(local_dir, f"{model_name}_{validation_name}"),
                    save_dir=os.path.join(save_dir, f"{model_name}_{validation_name}"),
                )

                if self.writer is not None:
                    if isinstance(obj, dict):
                        writer_dict = {
                            f"validation_{validation_name}_{model_name}_{key}": value
                            for key, value in obj.items()
                            if value is not None
                        }
                    elif obj is not None:
                        writer_dict = {f"validation_{validation_name}_{model_name}": obj}
                    else:
                        writer_dict = {}

                    if writer_dict:
                        self.writer.log(writer_dict, step=(self.iter + 1) // self.engine_config.ga_steps)
                comm.barrier()  # make sure all processes have finished validation before moving on to the next one

        for model_name, training in model_training_modes.items():
            self.models[model_name].train(training)
        validation_end_time = time.perf_counter()
        sstring = f"[Validation @ Steps {self.iter+1:07d}] | total_time: {validation_end_time - validation_start_time:.2f} s"

        # memory info
        memory_free, memory_total = torch.cuda.mem_get_info()
        memory_used = (memory_total - memory_free) / (1024 ** 3)  # in GB
        num_alloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
        sstring += f" | memory_used: {memory_used:.2f} GB"
        sstring += f" | num_alloc_retries: {num_alloc_retries}"

        logger.info(sstring)
        torch.cuda.empty_cache()
        gc.collect()
