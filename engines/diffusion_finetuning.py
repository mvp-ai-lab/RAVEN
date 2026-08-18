"""DiffusionFinetuning: the single engine of this project.

One project == one task == one engine file. launch.py instantiates
``DiffusionFinetuning(config, meta_model)`` and calls ``.run()``. Everything
heavy lives in ``common/`` (models/optim/data/persistence/logging/distributed);
everything algorithmic lives in the project meta model; verbatim-reusable
training mechanics (resume, optimize, EMA, metric/checkpoint plumbing) come
from ``common.engine.BaseEngine`` -- a method library, not an orchestration
framework: construction order, the train loop and the primitive sequence below
are all owned HERE. Base methods are public (``self.optimize()``); this file's
own internals keep the ``_`` prefix (``self._training_loop()``).

Meta-model contract (the ctx discipline)
----------------------------------------
Every algorithm primitive is called as ``ctx = meta.<primitive>(ctx)`` -- one
dict in, the same dict out (mutated). Rules:

* Each primitive documents its reads/writes keys; that IS its signature.
* Read with ``ctx["k"]`` (fail loud), never ``ctx.get("k", default)``.
* One writer per key: two primitives writing the same key is a design bug.
* ctx is STEP-LOCAL and volatile: built fresh each step, discarded after, never
  checkpointed. Anything that must survive a step lives on the engine and goes
  through the checkpoint state tree -- hiding cross-step state inside ctx (or
  inside the meta model) silently breaks bit-exact resume.

The engine seeds ctx with (fixed keys)::

    config      full CfgNode (read-only by convention)
    iter        current inner iteration (int)
    batch       the raw dataloader item
    rng         RandomState -- ALL step randomness derives from it; it is
                reconstructed from (seed, iter, sp_group) so a resumed run
                replays the identical stream. Never touch global RNG in a step.
    models      engine's name->module dict (long-lived, owned by engine)

and consumes after the primitives ran::

    loss        scalar tensor (graph attached) -- written by compute_loss
    metrics     dict[str, float] of extra scalars to aggregate + log (optional)

Primitive call sequence (one training step)::

    prepare_inputs   reads batch/rng/models  writes inputs (vae-encoded latents,
                     text embeddings, masks, data_info -- everything the model
                     needs, moved to device)
    sync_inputs      SP only: yields one sub-ctx per SP-group source batch
                     (unified-parallel input broadcast lives INSIDE the meta
                     model, engine just iterates). Group-off: yields ctx once.
    sample_timesteps reads inputs/models/rng/config  writes train_y plus timesteps
                     (scalar, per-chunk or per-frame -- DF/ITF policy is meta's)
    add_noise        reads inputs/timesteps/rng  writes noisy_latents (+noise)
    forward          reads models/inputs/train_y/noisy_latents/timesteps
                     writes pred (ITF: builds physical sequence + itf_ctx here)
    compute_loss     reads pred/inputs/...  writes loss (and metrics)

Config shape (engine-owned keys)::

    entry:
      module: engines.diffusion_finetuning
      class_name: DiffusionFinetuning
    meta_model:                                    # top-level: built by launch.py,
      module: ...                                  # passed into the engine ctor
      class_name: ...
    engine:
      training_steps: 1000000
      ga_steps: 1                                  # gradient accumulation
      up_size:
        train: 2
        validation: 8
      early_stop_hours: 3.85                       # optional runtime limit; omitted disables it
      early_stop_validate: false                   # run one validation when the runtime limit fires
      log_interval: 50                             # in optimizer steps
      save_start_step: 0
      save_interval: 1000
      val_start_step: 0
      val_interval: 0                              # 0 disables validation
      save_before_train: false
      val_before_train: false
      resume: auto                                 # auto | never | <int step>
      resume_dir: null                             # optional seed: another run's checkpoints dir or numeric step dir, used only when this run has none
      seed: 1019                                   # data/noise stream root
      clip_grad_norm: 1.0
      ema_decay:
        backbone: 0.9999                        # one decay per model with an EMA shadow
    validation:
      seed: 1
    distributed:
      up_size: [2, 8]

LR schedules are NOT engine config: an ``lr_scheduler`` node sits inside a
model's optimizer item (whole-optimizer) or inside one of its ``groups``
(single param_group) -- see common/lr_scheduler. The engine builds them via
``build_lr_schedulers`` and steps them as pure ``step(t)`` objects with t in
optimizer steps; nothing about them is checkpointed.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import torch

from common.data import build_dataloader
from common.distributed import ops
from common.engine import BaseEngine
from common.logging import get_logger
from common.lr_scheduler import build_lr_schedulers
from common.meter import AverageMeter, get_running_accumulator
from common.model import build_models
from common.optim import build_optimizers, checkpoint_models
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed

logger = get_logger()


class DiffusionFinetuning(BaseEngine):
    """Diffusion-denoising finetuning engine (DF and ITF ready via the meta model)."""

    def __init__(self, config: Any, meta_model: Any):
        self._start_time = time.monotonic()
        self.config = config
        self.engine_config = config.engine
        if "early_stop_hours" in self.engine_config:
            early_stop_hours = float(self.engine_config.early_stop_hours)
            if not early_stop_hours > 0:
                raise ValueError("engine.early_stop_hours must be greater than zero")
            self._early_stop_seconds = early_stop_hours * 60 * 60
        else:
            self._early_stop_seconds = None
        # Built by launch.py from config.meta_model (module/class_name), BEFORE the
        # engine exists: the meta model sees only the config at construction time;
        # models/rng/batch reach its primitives through ctx.
        self.meta = meta_model
        self.iter = 0

        # Register every project topology before model/FSDP construction.
        self.setup_unified_parallel()

        ga_steps = int(self.engine_config.ga_steps)
        train_up_size = self._train_up_size
        training_steps = int(self.engine_config.training_steps)
        save_interval = int(self.engine_config.save_interval)
        if (training_steps * ga_steps) % train_up_size != 0:
            raise ValueError("engine.training_steps * engine.ga_steps must be divisible by engine.up_size.train")
        if (save_interval * ga_steps) % train_up_size != 0:
            raise ValueError("engine.save_interval * engine.ga_steps must be divisible by engine.up_size.train")

        val_interval = int(self.engine_config.get("val_interval", 0))
        if val_interval > 0:
            if (val_interval * ga_steps) % train_up_size != 0:
                raise ValueError("engine.val_interval * engine.ga_steps must be divisible by engine.up_size.train")

        # -- build: models -> optimizers -> dataloader -> meta -------------------
        # Order matters: optimizers hold post-placement (sharded) parameters.
        self.models = build_models(config.models)
        self.optimizers = build_optimizers(self.models, config.models)
        self.lr_schedulers = build_lr_schedulers(self.optimizers, config.models)
        self.dataloader = build_dataloader(config.data, seed=int(self.engine_config.seed))

        # -- checkpoint state tree + resume --------------------------------------
        # Everything that must survive a restart, and nothing else. ctx never
        # appears here by design.
        self._runtime_state = {"iter": 0}
        self.state_tree = {
            "models": checkpoint_models(self.models, self.optimizers),
            "optimizers": self.optimizers,
            "dataloader": self.dataloader,
            "runtime": self._runtime_state,
            "accumulator": get_running_accumulator(),
        }
        eval_only = bool(self.engine_config.get("eval_only", False))
        if eval_only:
            # Eval consumes no data and takes no optimizer step, so neither state
            # is loaded. Restoring the dataloader snapshot would also tie the eval
            # to the corpus layout at save time -- shards since moved into a
            # held-out split invalidate the snapshot's resident row group -- for
            # state the eval never touches.
            del self.state_tree["dataloader"]
            del self.state_tree["optimizers"]
        self._resumed = self.maybe_resume()
        assert eval_only or self.iter == self.dataloader.consumed * self._train_up_size, (
            "resumed iter must equal dataloader.consumed * engine.up_size.train"
        )
        assert self.iter % ga_steps == 0, "resumed iter must align with engine.ga_steps"
        if not self._resumed:
            self.sync_ema_from_model()  # step-0 contract: EMA starts as an exact copy

    # -------------------------------------------------------------- the run --
    def run(self) -> None:
        cfg = self.engine_config
        ga_steps = int(cfg.ga_steps)
        up_size = self._train_up_size
        total_iters = int(cfg.training_steps) * ga_steps

        if cfg.get("eval_only", False):
            assert self._resumed, "engine.eval_only requires a resumed checkpoint (set engine.resume/resume_dir)"
            self.validate(step=self.resumed_step)
            return

        if cfg.get("save_before_train", False) and not self._resumed:
            self._runtime_state["iter"] = self.iter
            self.config.checkpointer.save(
                self.state_tree,
                step=self.iter // int(cfg.ga_steps),
            )
        if cfg.get("val_before_train", False) and not self._resumed:
            self.validate(step=0)

        logger.info(
            "Training: iters %d -> %d (ga_steps=%d, sp=%d x %d groups)",
            self.iter, total_iters, ga_steps, up_size, self.num_sp_groups,
        )

        step_timer = time.perf_counter()
        meter = AverageMeter()
        while self.iter < total_iters:
            # timer/* split (per observation, averaged over the log window):
            # dataload and prepare tick once per dataloader batch, train once per
            # iter; timer/iter (in log_metrics) is the all-in wall clock.
            tick = time.perf_counter()
            batch = next(self.dataloader)
            meter.put_scalar("timer/dataload", time.perf_counter() - tick)

            # Per-rank stream: pre-sync work (augmentation etc.) may differ per rank.
            ctx: dict[str, Any] = {
                "config": self.config,
                "iter": self.iter,
                "batch": batch,
                "models": self.models,
                "rng": RandomState(seed=combine_seed(cfg.seed, "prepare", self.iter, self.sp_group_id)),
            }
            tick = time.perf_counter()
            ctx = self.meta.prepare_inputs(ctx)
            meter.put_scalar("timer/prepare", time.perf_counter() - tick)

            # SP input broadcast is the meta model's business: one sub-ctx per
            # SP-group source batch (group-off: yields exactly once). Group
            # mates share a stream -- rng is re-derived per sub-ctx below.
            for up_index, sub_ctx in enumerate(self._iter_synced(ctx), start=1):
                if self.iter >= total_iters:
                    continue
                sub_ctx["iter"] = self.iter
                sub_ctx["rng"] = RandomState(
                    seed=combine_seed(cfg.seed, "step", sub_ctx["iter"], self.sp_group_id)
                )

                # Pure step(t) objects from common/lr_scheduler -- resume-safe by
                # construction. t is the OPTIMIZER step (ga microsteps collapse
                # onto one t), so warmup_t etc. are counted in real update steps.
                for scheduler in self.lr_schedulers.values():
                    scheduler.step(self.iter // ga_steps)
                tick = time.perf_counter()
                log_dict = self._training_loop(sub_ctx)
                meter.put_scalar("timer/train", time.perf_counter() - tick)

                meter.update(log_dict)

                # One clean node per training cycle: the final UP yield is where
                # log/save/val/runtime-stop all fire -- checkpoint/resume
                # invariants hold only with all synchronized inputs consumed.
                if up_index == self._train_up_size:
                    step_timer, stop = self.process_due(
                        meter, step_timer,
                        total_iters=total_iters,
                    )
                    if stop:
                        return

                self.iter += 1

        logger.info("Training complete at iter=%d", self.iter)

    def _iter_synced(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield from self.meta.sync_inputs(ctx)

    # ------------------------------------------------------------- one step --
    def _training_loop(self, ctx: dict[str, Any]) -> dict[str, float]:
        cfg = self.engine_config
        ga = int(cfg.ga_steps)
        is_sync_iter = (self.iter + 1) % ga == 0
        self.set_gradient_sync(True)  # sync every ga-microstep: FSDP2 reduce-scatters
        # each backward, keeping grads sharded (no-sync accumulation costs memory)

        ctx = self.meta.sample_timesteps(ctx)
        ctx = self.meta.add_noise(ctx)
        ctx = self.meta.forward(ctx)
        ctx = self.meta.compute_loss(ctx)

        loss = ctx["loss"]
        log_dict = {"train/loss": float(loss.detach().item())}
        log_dict.update(ctx.get("metrics", {}))
        self.backward(loss / ga)

        if is_sync_iter:
            log_dict.update(self.optimize())
        return log_dict
