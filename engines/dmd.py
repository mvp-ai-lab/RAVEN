"""DistributionMatchingDistillation: the single engine of this project.

One project == one task == one engine file. launch.py instantiates
``DistributionMatchingDistillation(config, meta_model)`` and calls ``.run()``.
Everything heavy lives in ``common/`` (models/optim/data/persistence/logging/
distributed); everything algorithmic lives in the project meta model;
verbatim-reusable training mechanics (resume, optimize, EMA, metric/checkpoint
plumbing) come from ``common.engine.BaseEngine`` -- a method library, not an
orchestration framework: construction order, the train loop and the primitive
sequence below are all owned HERE. Base methods are public (``self.optimize()``);
this file's own internals keep the ``_`` prefix (``self._fake_backward()``).

DMD (distribution matching distillation) = one no-grad generator rollout per
micro-iteration plus two TTUR-gated optimization phases per logical optimizer
window. A window first accumulates all FAKE microbatches and steps the fake
score model once, then accumulates all GEN microbatches against that same
post-step fake model and steps the generator once. This phase barrier preserves
FAKE-before-GEN ordering while supporting ``ga_steps >= 1``. During GEN the meta
model's context manages algorithm-specific model state across forward, backward,
and optimization. Unified parallel optionally runs one synchronized source
batch per SP-group rank; validation runs only at logical optimizer-step boundaries.

Meta-model contract (the ctx discipline)
----------------------------------------
Every algorithm primitive is called as ``ctx = meta.<primitive>(ctx)`` -- one
dict in, the same dict out (mutated). Rules:

* Each primitive documents its reads/writes keys; that IS its signature.
* Read with ``ctx["k"]`` (fail loud), never ``ctx.get("k", default)``.
* One writer per key: two primitives writing the same key is a design bug.
* Phase ctx is MICRO-ITERATION-LOCAL and volatile: built fresh for one phase and
  discarded after backward. A detached base rollout payload may live until the
  end of the current offline cycle, but is never checkpointed. Anything that
  must survive a save boundary lives on the engine and goes through the
  checkpoint state tree -- hiding persistent state inside ctx (or inside the
  meta model) silently breaks bit-exact resume.

The engine seeds ctx with (fixed keys)::

    config      full CfgNode (read-only by convention)
    iter        current 0-based micro-iteration (int)
    batch       the raw dataloader item
    models      engine's name->module dict (long-lived, owned by engine)
    rng         RandomState -- ALL step randomness derives from it; it is
                re-seeded per phase from (seed, tag, iter, rank-or-SP-group) with
                tags "prepare"/"rollout"/"fake"/"gen" so a resumed run replays the
                identical stream. Never touch global RNG in a step. Per-phase
                re-seeding means draw-order equivalence with the reference
                implementation holds WITHIN each phase; runs are not
                bit-identical to the old single-stream implementation
                (deliberate framework decision).

Primitive call sequence (each micro-iteration; writer -> keys)::

    prepare_inputs   writes inputs, neg_inputs
    sync_inputs      SP only: yields one synchronized input per SP-group source rank
    rollout          writes rollout_x0s, trajectory_xts       (no-grad)
    -- FAKE phase (gated once per logical window; all FAKE precede all GEN) --
    prepare_fake     writes fake_timesteps, fake_noises, fake_noisy_latents, fake_inputs
    fake_forward     writes fake_pred
    fake_loss        writes fake_loss  (a scalar, or a sequence of scalars that
                     are backwarded one per GraphTask -- see _fake_backward)
    -- GEN phase (gated once per logical window) --
    prepare_gen      writes gen_timesteps, gen_index, gen_xts, score_timesteps, gen_inputs
    gen_forward      writes gen_pred                          (graph attached)
    score            writes gen_x0s, fake_score_x0s, real_score_x0s
    gen_loss         writes gen_loss

Validation receives its own project-owned ctx with ``config``, ``models`` and
the explicit logical optimizer ``step``.

``fake_metrics``/``gen_metrics`` are OPTIONAL keys the engine reads with a
default -- the current meta model does not write them; detailed metrics flow
through the running meter singletons instead.

Expected configuration shape::

    entry:
      module: engines.dmd
      class_name: DistributionMatchingDistillation
    meta_model:                                # top-level: built by launch.py,
      module: ...                              # passed into the engine ctor
      class_name: ...
    engine:
      training_steps: 300                      # logical optimizer windows
      ga_steps: 1                              # micro-iterations per window (>= 1)
      offline: 1                               # rollout windows per short-lived pool
      seed: 20001019                           # data/noise stream root
      log_interval: 5                          # in optimizer steps
      save_interval: 20
      save_start_step: 0
      save_before_train: false
      val_before_train: false
      val_interval: 0                         # 0 disables interval validation
      val_start_step: 0
      resume: auto                             # auto | never | <int step>
      resume_dir: null                         # optional seed: another run's checkpoints dir or numeric step dir, used only when this run has none
      early_stop_hours: 3.85                   # optional runtime limit; omitted disables it
      clip_grad_norm: 1000.0
      ema_decay:
        backbone: 0.9675                       # one decay per model with an EMA shadow
      fake_update_interval: 1                  # TTUR: run FAKE every N windows
      gen_update_interval: 2                   # TTUR: run GEN every N windows
      fake_step_models: [model_name]           # optimize() targets of the FAKE phase
      gen_step_models: [backbone]              # optimize() targets of the GEN phase
      up_size:
        train: 2
        validation: 8
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
from typing import Any

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


class DistributionMatchingDistillation(BaseEngine):
    """DMD engine: no-grad rollout + TTUR-gated fake/gen optimization phases."""

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

        cfg = self.engine_config
        ga = int(cfg.ga_steps)
        fake_interval = cfg.fake_update_interval
        gen_interval = cfg.gen_update_interval
        assert fake_interval == 1 or gen_interval == 1
        val_interval = int(cfg.get("val_interval", 0))
        val_before_train = cfg.get("val_before_train", False)
        self._val_interval = val_interval
        self._val_before_train = val_before_train

        training_steps = int(cfg.training_steps)
        save_interval = int(cfg.save_interval)
        if (training_steps * ga) % self._train_up_size != 0:
            raise ValueError("engine.training_steps * engine.ga_steps must be divisible by engine.up_size.train")
        if (save_interval * ga) % self._train_up_size != 0:
            raise ValueError("engine.save_interval * engine.ga_steps must be divisible by engine.up_size.train")
        if val_interval > 0 and (val_interval * ga) % self._train_up_size != 0:
            raise ValueError("engine.val_interval * engine.ga_steps must be divisible by engine.up_size.train")

        # -- build: models -> optimizers -> dataloader ----------------------------
        # Order matters: optimizers hold post-placement (sharded) parameters.
        self.models = build_models(config.models)
        self.optimizers = build_optimizers(self.models, config.models)
        self.lr_schedulers = build_lr_schedulers(self.optimizers, config.models)
        self.dataloader = build_dataloader(config.data, seed=int(self.engine_config.seed))
        self._synced_inputs_pool: list[dict[str, Any]] = []

        # -- checkpoint state tree + resume ---------------------------------------
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
        self._resumed = self.maybe_resume()
        if self._resumed:
            assert self.iter == self.resumed_step * ga, (
                f"checkpoint step {self.resumed_step} with ga_steps={ga} requires runtime iter "
                f"{self.resumed_step * ga}, got {self.iter}"
            )
            assert self.iter % ga == 0, f"resumed iter={self.iter} is not aligned to ga_steps={ga}"
            assert self.dataloader.consumed * self._train_up_size == self.iter, (
                f"resume dataloader mismatch: consumed={self.dataloader.consumed} "
                f"up_size={self._train_up_size} iter={self.iter}"
            )
        if not self._resumed:
            self.sync_ema_from_model()  # step-0 contract: EMA starts as an exact copy

    # -------------------------------------------------------------- the run --
    def run(self) -> None:
        cfg = self.engine_config
        ga = int(cfg.ga_steps)
        offline = int(getattr(cfg, "offline", 1))
        fake_interval = int(cfg.fake_update_interval)
        gen_interval = int(cfg.gen_update_interval)
        save_interval = int(cfg.save_interval)
        val_interval = int(self._val_interval)
        total_steps = int(cfg.training_steps)
        up_size = self._train_up_size

        assert offline >= 1, f"offline must be >= 1, got {offline}"
        if offline > 1:
            assert total_steps % offline == 0, (
                f"training_steps={total_steps} must be divisible by offline={offline}"
            )
            assert save_interval % offline == 0, (
                f"save_interval={save_interval} must be divisible by offline={offline}"
            )
            assert val_interval == 0 or val_interval % offline == 0, (
                f"val_interval={val_interval} must be 0 or divisible by offline={offline}"
            )
            assert (self.iter // ga) % offline == 0, (
                f"step={self.iter // ga} is not aligned to offline={offline}"
            )
            ttur_period = max(fake_interval, gen_interval)
            assert offline % ttur_period == 0, (
                f"offline={offline} must be divisible by TTUR period={ttur_period}"
            )

        total_iters = total_steps * ga
        logger.info(
            "Training: micro-iters %d -> %d (ga_steps=%d, offline=%d, fake_update_interval=%d, "
            "gen_update_interval=%d, sp=%d x %d groups)",
            self.iter, total_iters, ga, offline, fake_interval, gen_interval, up_size, self.num_sp_groups,
        )

        if cfg.save_before_train and not self._resumed:
            self._runtime_state["iter"] = self.iter
            self.config.checkpointer.save(self.state_tree, step=self.iter // ga)

        if self._val_before_train and not self._resumed:
            self.validate(step=0)

        step_timer = time.perf_counter()
        meter = AverageMeter()
        rank = ops.get_rank()
        while self.iter < total_iters:
            cycle_start_iter = self.iter
            cycle_start_step = cycle_start_iter // ga
            pool = self._refill_pool(cycle_start_iter, offline, ga, rank, self.sp_group_id, meter)

            for cycle_offset in range(offline):
                logical_step = cycle_start_step + cycle_offset
                completed_step = logical_step + 1
                do_fake = completed_step % fake_interval == 0
                do_gen = completed_step % gen_interval == 0
                base_contexts = pool[:ga]
                del pool[:ga]
                for g, base_ctx in enumerate(base_contexts):
                    base_ctx["iter"] = cycle_start_iter + cycle_offset * ga + g

                for scheduler in self.lr_schedulers.values():
                    scheduler.step(logical_step)

                if do_fake:
                    for base_ctx in base_contexts:
                        self.iter = int(base_ctx["iter"])
                        tick = time.perf_counter()
                        # FSDP2 no-sync retains full gradients on every rank.
                        meter.update(self._fake_backward(base_ctx, self.sp_group_id, sync=True))
                        meter.put_scalar("timer/fake", time.perf_counter() - tick)
                    meter.update(self.optimize(names=list(cfg.fake_step_models)))

                if do_gen:
                    with self.meta.gen_model_context(self.models):
                        for base_ctx in base_contexts:
                            self.iter = int(base_ctx["iter"])
                            tick = time.perf_counter()
                            # FSDP2 no-sync retains full gradients on every rank.
                            meter.update(self._gen_backward(base_ctx, self.sp_group_id, sync=True))
                            meter.put_scalar("timer/gen", time.perf_counter() - tick)
                        meter.update(self.optimize(names=list(cfg.gen_step_models)))

                self.iter = cycle_start_iter + (cycle_offset + 1) * ga - 1
                del base_contexts, base_ctx

                # One clean node per offline cycle: the final cycle with the sync
                # pool drained is where log/save/val/runtime-stop all fire --
                # checkpoint/resume invariants hold only there.
                if cycle_offset + 1 == offline and not self._synced_inputs_pool:
                    step_timer, stop = self.process_due(
                        meter, step_timer,
                        total_iters=total_iters,
                    )
                    if stop:
                        return

                self.iter += 1

            del pool

        logger.info("Training complete at iter=%d", self.iter)

    def _refill_pool(
        self,
        cycle_start_iter: int,
        offline: int,
        ga: int,
        rank: int,
        sp_group_id: int,
        meter: AverageMeter,
    ) -> list[dict[str, Any]]:
        """Prepare one flat pool of complete rollout payloads without advancing iter."""
        cfg = self.engine_config
        pool: list[dict[str, Any]] = []

        target_size = offline * ga
        while len(pool) < target_size:
            source_micro_iter = cycle_start_iter + len(pool)
            if not self._synced_inputs_pool:
                tick = time.perf_counter()
                batch = next(self.dataloader)
                meter.put_scalar("timer/dataload", time.perf_counter() - tick)
                ctx: dict[str, Any] = {
                    "config": self.config,
                    "iter": source_micro_iter,
                    "batch": batch,
                    "models": self.models,
                    "rng": RandomState(
                        seed=combine_seed(cfg.seed, "prepare", source_micro_iter, rank)
                    ),
                }
                tick = time.perf_counter()
                ctx = self.meta.prepare_inputs(ctx)
                meter.put_scalar("timer/prepare", time.perf_counter() - tick)
                if self._train_up_size <= 1:
                    self._synced_inputs_pool.append(
                        {"inputs": ctx["inputs"], "neg_inputs": ctx["neg_inputs"]}
                    )
                else:
                    self._synced_inputs_pool.extend(
                        {
                            "inputs": synced_ctx["inputs"],
                            "neg_inputs": synced_ctx["neg_inputs"],
                        }
                        for synced_ctx in self.meta.sync_inputs(ctx)
                    )
                assert len(self._synced_inputs_pool) == self._train_up_size, (
                    f"sync_inputs yielded {len(self._synced_inputs_pool)}, expected {self._train_up_size}"
                )
                del batch, ctx

            sub_ctx = self._synced_inputs_pool.pop(0)
            sub_ctx["iter"] = source_micro_iter
            sub_ctx["config"] = self.config
            sub_ctx["models"] = self.models
            sub_ctx["rng"] = RandomState(
                seed=combine_seed(cfg.seed, "rollout", source_micro_iter, sp_group_id)
            )
            tick = time.perf_counter()
            sub_ctx = self.meta.rollout(sub_ctx)
            meter.put_scalar("timer/rollout", time.perf_counter() - tick)
            payload = {
                "iter": source_micro_iter,
                "inputs": sub_ctx["inputs"],
                "neg_inputs": sub_ctx["neg_inputs"],
                "rollout_x0s": sub_ctx["rollout_x0s"],
                "trajectory_xts": sub_ctx["trajectory_xts"],
            }
            pool.append(payload)
            del sub_ctx, payload

        if offline > 1:
            RandomState(
                seed=combine_seed(cfg.seed, "offline_order", cycle_start_iter, sp_group_id)
            ).python_generator.shuffle(pool)
        return pool

    # ---------------------------------------------------------- TTUR phases --
    def _fake_backward(self, base_ctx: dict[str, Any], sp_group_id: int, sync: bool) -> dict[str, float]:
        """Run one scaled FAKE backward without stepping its optimizer."""
        cfg = self.engine_config
        micro_iter = int(base_ctx["iter"])
        self.set_gradient_sync(sync)
        ctx = dict(base_ctx)
        ctx["config"] = self.config
        ctx["models"] = self.models
        ctx["rng"] = RandomState(seed=combine_seed(cfg.seed, "fake", micro_iter, sp_group_id))

        ctx = self.meta.prepare_fake(ctx)
        ctx = self.meta.fake_forward(ctx)
        ctx = self.meta.fake_loss(ctx)

        # ``fake_loss`` is a scalar, or a sequence of scalars to backward one at
        # a time. The sequence form is not cosmetic and not gradient
        # accumulation: it controls how many autograd GraphTasks the FAKE step
        # runs, which is what bounds its peak memory.
        #
        # A DMD2 FAKE step forwards fake_model more than once (score MSE, then
        # the discriminator's classification), and FSDP2 reuses ONE unsharded
        # Parameter object across those forwards -- `torch.ops.fsdp.set_` swaps
        # storage, not the TensorImpl. Summing the terms and backwarding once
        # therefore gives every parameter's AccumulateGrad a dependency count
        # equal to the number of forwards, and the engine parks each block's
        # gradient in an autograd InputBuffer until the LAST subgraph reaches
        # it. Nothing is lost -- but nothing is released either: measured at
        # 1231 MiB per block, ~60 GiB across a 33 B trunk, and FSDP's
        # post_backward runs in between finding `.grad` empty and reduce-
        # scattering nothing (jobs 15439786, 15440105, 15442370).
        #
        # One backward per term gives each its own GraphTask, so every
        # dependency count is 1, AccumulateGrad fires immediately, and each
        # block's gradient is reduce-scattered as it is produced. Gradients
        # still accumulate across terms -- FSDP2's foreach_reduce adds into
        # `sharded_param.grad` rather than overwriting -- so the parameter
        # update is unchanged. The cost is one reduce-scatter pass per term.
        loss = ctx["fake_loss"]
        terms = tuple(loss) if isinstance(loss, (list, tuple)) else (loss,)
        total = sum(float(term.detach()) for term in terms)
        for term in terms:
            self.backward(term / int(cfg.ga_steps))

        log_dict = {"train/fake_loss": total}
        log_dict.update(ctx.get("fake_metrics", {}))
        del loss, terms, ctx
        return log_dict

    def _gen_backward(self, base_ctx: dict[str, Any], sp_group_id: int, sync: bool) -> dict[str, float]:
        """Run one scaled GEN backward against the post-barrier fake model."""
        cfg = self.engine_config
        micro_iter = int(base_ctx["iter"])
        self.set_gradient_sync(sync)
        ctx = dict(base_ctx)
        ctx["config"] = self.config
        ctx["models"] = self.models
        ctx["rng"] = RandomState(seed=combine_seed(cfg.seed, "gen", micro_iter, sp_group_id))

        ctx = self.meta.prepare_gen(ctx)
        ctx = self.meta.gen_forward(ctx)
        ctx = self.meta.score(ctx)
        ctx = self.meta.gen_loss(ctx)

        loss = ctx["gen_loss"]
        self.backward(loss / int(cfg.ga_steps))

        log_dict = {"train/gen_loss": float(loss.detach().item())}
        log_dict.update(ctx.get("gen_metrics", {}))
        del loss, ctx
        return log_dict
