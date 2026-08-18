"""TrajectorySegmentedConsistencyDistillation: the single TSCD engine.

One project == one task == one engine file. launch.py instantiates
``TrajectorySegmentedConsistencyDistillation(config, meta_model)`` and calls
``.run()``. Everything heavy lives in ``common/`` (models/optim/data/
persistence/logging/distributed); everything algorithmic lives in the project
meta model; verbatim-reusable training mechanics (resume, optimize, EMA,
metric/checkpoint plumbing) come from ``common.engine.BaseEngine`` -- a method
library, not an orchestration framework: construction order, the train loop and
the primitive sequence below are all owned HERE. Base methods are public
(``self.optimize()``); this file's own internals keep the ``_`` prefix
(``self._training_loop()``).

TSCD trains one online model while a frozen teacher and the online model's EMA
shadow supply no-grad auxiliary computations. The engine owns only scheduling:
it contains no diffusion schedules, samplers, timestep grids or solver math.
The meta-model primitives collectively implement the following algorithm.

A descending shifted grid ``t_0 > t_1 > ... > t_{N-1}`` over ``[0, T]`` is
split into ``S`` segments of ``L = N // S`` grid indices. Segment boundaries
are at indices ``L, 2L, ..., S*L``; an index ``>= N`` maps to the clean
boundary ``t = 0``. For one sample:

1. sample grid index ``i`` and take ``t = t_i``;
2. take the adjacent grid point ``t' = t_{i+1}`` as the ODE-step target;
3. take ``t_b = t_{(i//L + 1) * L}`` as the clean-side segment boundary;
4. draw ``s ~ (t_b, t']`` as ``s = t_b + u*(t' - t_b)`` with
   ``u = 1 - U[0,1)``;
5. form ``x_t = A(t) x_0 + B(t) x_T`` from one noise draw;
6. run one online-student forward on ``(x_t, t)`` with its graph attached;
7. run the teacher on ``(x_t, t)``, then exactly one deterministic ODE/DDIM
   step from ``x_t`` to ``x_{t'}``, under no-grad;
8. run the EMA backbone on ``(x_{t'}, t')`` under no-grad for the consistency
   target;
9. project both predictions to a common comparison time and apply
   stop-gradient MSE.

There are two distinct "end" times: the teacher integrates only to the adjacent
point ``t'``; the loss compares at ``s`` (or at ``t_b``). The segment boundary
is a comparison target, never the teacher integration target. Conflating these
is the classic TSCD porting bug. The reference algorithm has no ``c_skip`` /
``c_out`` boundary-condition scalings, no Huber loss, and no automatic
8->4->2->1 segment scheduler. Stage shrinking is a separate-run configuration
choice, not engine behavior.

``S`` interpolates between two known algorithms. ``S = 1`` puts every grid
index in one segment, so ``t_b`` is the clean end of the grid and the
self-consistency constraint becomes "every ``t`` maps to the trajectory
endpoint" -- plain consistency distillation. ``S = N`` makes ``t_b = t'``, the
boundary and the integration target coincide, and the objective degenerates to
step-wise denoising distillation. Everything between is TSCD proper.

Classifier-free guidance is deliberately absent from the description above.
Whether the teacher runs an explicit two-pass CFG, whether the student is
conditioned on a guidance scale, and whether negative conditioning exists at
all are meta-model decisions -- some backbones ship already guidance-distilled
and have no unconditional branch to call. The engine cannot observe the
difference: it only sequences primitives, and all of this is internal to
``student_forward`` and ``solver_step``.

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
    models      engine's name->module dict (long-lived, owned by engine)
    rng         RandomState -- ALL step randomness derives from it; it is
                reconstructed from (seed, iter, sp_group) so a resumed run
                replays the identical stream. Never touch global RNG in a step.

and consumes after the primitives ran::

    loss        scalar tensor (graph attached) -- written by compute_loss
    metrics     dict[str, float] of extra scalars to aggregate + log (optional)

Primitive call sequence (one training step)::

    prepare_inputs   writes whatever the later primitives consume -- latents,
                     conditioning, and any negative conditioning the meta model
                     needs. The engine reads none of it.
    sync_inputs      SP only: yields one sub-ctx per SP-group source batch
                     (unified-parallel input broadcast lives INSIDE the meta
                     model, engine just iterates). Group-off: yields ctx once.
    sample_segment   writes start_timesteps, step_timesteps,
                     boundary_timesteps, target_timesteps
    add_noise        writes noises, noisy_latents
    student_forward  writes student_pred (graph attached)
    solver_step      writes solver_xts (no-grad; teacher forward plus exactly
                     one ODE/DDIM step to step_timesteps, never
                     boundary_timesteps)
    target_forward   writes target_pred (no-grad; EMA backbone at
                     (solver_xts, step_timesteps))
    compute_loss     writes loss (and optional metrics)

``solver_step`` and ``target_forward`` stay separate because they run different
models (``tea_model`` versus ``backbone_ema``). This split lets a project
overlap EMA FSDP unsharding with the teacher forward, and lets an ablation swap
the solver or remove the EMA half without changing the other primitive.

Config ownership and expected shape
-----------------------------------
Algorithm hyperparameters belong to ``meta_model`` -- ``num_segments``,
``loss_type``, and whatever else the chosen formulation needs -- while the
schedules, samplers and timestep grids they drive are built by
``common.diffusion`` from the top-level ``diffusion`` node. The engine reads neither. This intentionally
differs from the earlier implementation, where ``num_segments`` and
``loss_type`` lived in engine config. ``engine`` contains only generic training
mechanics::

    entry:
      module: engines.tscd
      class_name: TrajectorySegmentedConsistencyDistillation
    meta_model:                                    # built by launch.py
      module: ...
      class_name: ...
      num_segments: 4                              # splits the grid below into
      loss_type: x_t                               # this many segments
      ...                                          # plus whatever else the
                                                   # formulation needs
    diffusion:                                     # built by build_diffusion;
      sampling_timesteps: ...                      # a sampler named <p>_sampler
      schedule: ...                                # is handed the schedule
      tea_schedule: ...                            # named <p>_schedule
      tea_sampler: ...                             # (DDIM: the one ODE step)
    models:
      <trained_name>:
        ema: ...                                   # builds <trained_name>_ema
    engine:
      training_steps: 1000000
      ga_steps: 1                                  # gradient accumulation
      up_size:
        train: 2
        validation: 8
      early_stop_hours: 3.85                       # optional; omitted disables it
      early_stop_validate: false
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
        <trained_name>: 0.9999                     # required TSCD target update
    validation:
      seed: 1
    distributed:
      up_size: [2, 8]

The trained model's EMA is load-bearing: it produces ``target_pred``, rather
than serving only as an optional evaluation shadow. Construction therefore
requires an ``engine.ema_decay`` entry for the optimizer-owning model, and a
fresh run synchronizes its EMA shell from the online model before step 0.

LR schedules are NOT engine config: an ``lr_scheduler`` node sits inside a
model's optimizer item (whole-optimizer) or inside one of its ``groups``
(single param_group) -- see common/lr_scheduler. The engine builds them via
``build_lr_schedulers`` and steps them as pure ``step(t)`` objects with t in
optimizer steps; nothing about them is checkpointed.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from common.data import build_dataloader
from common.engine import BaseEngine
from common.logging import get_logger
from common.lr_scheduler import build_lr_schedulers
from common.meter import AverageMeter, get_running_accumulator
from common.model import build_models
from common.optim import build_optimizers, checkpoint_models
from common.seed import RandomState, combine_seed

logger = get_logger()


class TrajectorySegmentedConsistencyDistillation(BaseEngine):
    """TSCD engine: one student backward with teacher and EMA target forwards."""

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
        trained_name, = self.optimizers
        if trained_name not in self.engine_config.get("ema_decay", {}):
            raise ValueError(
                f"TSCD needs an EMA target network; set engine.ema_decay.{trained_name}"
            )
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
        self._resumed = self.maybe_resume()
        assert self.iter == self.dataloader.consumed * self._train_up_size, (
            "resumed iter must equal dataloader.consumed * engine.up_size.train"
        )
        assert self.iter % ga_steps == 0, "resumed iter must align with engine.ga_steps"
        if not self._resumed:
            # TSCD trains against this target, so self-consistency requires an
            # exact online-model copy rather than the EMA shell's random init.
            self.sync_ema_from_model()

    # -------------------------------------------------------------- the run --
    def run(self) -> None:
        cfg = self.engine_config
        ga_steps = int(cfg.ga_steps)
        up_size = self._train_up_size
        total_iters = int(cfg.training_steps) * ga_steps

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

        ctx = self.meta.sample_segment(ctx)
        ctx = self.meta.add_noise(ctx)
        ctx = self.meta.student_forward(ctx)
        ctx = self.meta.solver_step(ctx)
        ctx = self.meta.target_forward(ctx)
        ctx = self.meta.compute_loss(ctx)

        loss = ctx["loss"]
        log_dict = {"train/loss": float(loss.detach().item())}
        log_dict.update(ctx.get("metrics", {}))
        self.backward(loss / ga)

        if is_sync_iter:
            log_dict.update(self.optimize())
        return log_dict
