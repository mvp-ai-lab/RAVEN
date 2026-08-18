"""GroupRelativePolicyOptimization: the single engine of this project.

One project == one task == one engine file. launch.py instantiates
``GroupRelativePolicyOptimization(config, meta_model)`` and calls ``.run()``.
Everything heavy lives in ``common/`` (models/optim/data/persistence/logging/
distributed); everything algorithmic lives in the project meta model;
verbatim-reusable training mechanics (resume, optimize, EMA, metric/checkpoint
plumbing) come from ``common.engine.BaseEngine`` -- a method library, not an
orchestration framework: construction order, the train loop and the primitive
sequence below are all owned HERE. Base methods are public (``self.optimize()``);
this file's own internals keep the ``_`` prefix (``self._policy_phase()``).

CM-GRPO (consistency-model GRPO), ported from RAVEN's
``project/engines/group_relative_policy_optimization.py``: a no-grad GROUP
rollout phase (``group_size`` trajectories per prompt batch, scored by the
reward model zoo) feeds a policy phase that walks the shuffled
(group, timestep) slice grid. The engine owns reward selection, advantage
bookkeeping (optionally via the cross-rank ``PerPromptStatTracker``), the group
split and the slice schedule; the meta model owns the rollout/reward/forward/
loss primitives. There is no sequence parallel.

Offline x GA structure (deliberate RAVEN divergence, mirroring the DMD project)
------------------------------------------------------------------------------
RAVEN's slice machinery (``slices_per_step`` micro-batching, ``optim_steps``
multi-step chunking, mutually exclusive with its outer ``ga_steps``) is
re-expressed as TWO divisibility knobs over ONE rollout's slice grid ``S``
(``N = group_size`` with ``random_policy_timestep``, else
``group_size x num_policy_steps``):

* ``offline`` (RAVEN ``optim_steps``): optimizer steps AMORTIZED over one group
  rollout. Requires ``offline | N``; each optimizer step trains on a disjoint
  shuffled chunk of ``C = N / offline`` slices. ``training_steps`` counts
  OPTIMIZER STEPS; a fresh rollout happens only every ``offline`` optimizer
  steps, so total rollouts = ``training_steps / offline`` (hence the assert
  ``training_steps % offline == 0``). Later chunks are trained after earlier
  chunks' updates -- increasingly off-policy, hence "offline".
* ``ga_steps`` (REPLACES RAVEN ``slices_per_step``): backwards accumulated per
  optimizer step. Requires ``ga_steps | C``; each backward merges
  ``k = C / ga_steps = N / (offline * ga_steps)`` slices into one packed
  forward and is scaled by ``1 / ga_steps`` -- identical to RAVEN's
  ``len(sub) / len(chunk)`` formula. The upstream geometry
  (N=4, k=2, one step) is ``offline: 1, ga_steps: 2``, bit-for-bit.
  Within one rollout all slices share identical token shapes (same prompt
  batch, same seqlens), so this knob cancels EXACTLY in the loss
  normalization -- it is pure memory/throughput batching (fp reassociation
  aside).

Step accounting is fully delegated to ``common.engine.BaseEngine`` by adopting
its own convention: ``iter`` counts BACKWARDS (dmd's "micro-iterations") and
``ga_steps`` keeps EXACTLY common's meaning (backwards per optimizer step).
Then common's ``save()`` (folder step ``(iter+1)//ga`` = optimizer steps,
matching RAVEN's checkpoint naming), ``log_metrics()``, ``at_interval()`` and
``maybe_resume()`` all hold verbatim -- no overrides. Checkpoints land only at
grid boundaries (``iter % (offline*ga) == 0`` after a completed backward,
guaranteed by ``save_interval % offline == 0``), keeping the resume invariants
``iter == resumed_step * ga`` and ``dataloader.consumed ==
iter // (offline * ga)`` with zero policy-phase state in the checkpoint.
FSDP2 gradient sync opens on the last backward of each optimizer step.

ctx discipline -- identical to the sibling DMD project: every meta primitive is
``ctx = meta.<primitive>(ctx)``, one dict in, the same dict out; read with
``ctx["k"]`` (fail loud); one writer per key; nothing persistent hides in ctx.

The engine seeds ctx with (fixed keys): ``config``, ``iter`` (the rollout
index, ``iter // (offline * ga_steps)``), ``batch``, ``models``, ``rng``. ALL step
randomness derives from per-phase re-seeded RandomStates with tags
"prepare"/"rollout"/"policy" keyed by the rollout index, so a resumed run
replays the identical stream (deliberate framework decision: draw-order
equivalence with RAVEN holds WITHIN each phase; runs are not bit-identical to
RAVEN's single-stream RNG).

Primitive call sequence (each rollout, once per ``offline`` optimizer steps;
writer -> keys)::

    prepare_inputs   writes inputs, neg_inputs                          (meta)
    rollout          writes infer_rollouts                              (meta, no-grad)
    -- engine: reward selection -> advantages -> group split -> slice schedule --
    prepare_policy   writes pos_step_inputs, neg_step_inputs, next_xt,
                     step_s, slice_advantages, slice_step_indices,
                     num_policy_steps                                   (meta)
    policy_forward   writes policy_pred (, ref_pred)                    (meta, graph)
    policy_loss      writes policy_loss                                 (meta)

Expected configuration shape::

    entry:
      module: engines.grpo
      class_name: GroupRelativePolicyOptimization
    meta_model:                                # top-level: built by launch.py,
      module: ...                              # passed into the engine ctor
      class_name: ...
    engine:
      training_steps: 160                      # optimizer steps (must be a multiple of offline)
      ga_steps: 2                              # backwards per optimizer step (RAVEN slices_per_step)
      offline: 1                               # optimizer steps amortized per rollout (RAVEN optim_steps)
      seed: 20001019                           # data/noise stream root
      log_interval: 10                         # in optimizer steps
      save_interval: 80                        # must be a multiple of offline
      save_start_step: 0
      save_before_train: false
      val_before_train: false
      val_interval: 80                         # 0 disables; must be a multiple of offline
      val_start_step: 0
      resume: auto                             # auto | never | <int step>
      resume_dir: null                         # optional seed: another run's checkpoints dir or numeric step dir, used only when this run has none
      clip_grad_norm: 1000.0
      ema_decay:
        backbone: 0.9675
      policy_step_models: [backbone]           # optimize() targets of the policy phase
      # -- GRPO knobs (RAVEN GroupRelativePolicyOptimizationConfig) --
      group_size: 4                            # rollouts per prompt batch (> 1)
      groups_per_infer: 4                      # rollouts merged into one infer pass
      reward_field: {raft_RAFT: 0.35, ...}     # str | list | dict of weights
      reward_aggregation: normalize_then_sum   # or sum_then_normalize
      adv_clip_max: 5.0
      reverse_advantage: false
      reverse_policy_advantage: false
      beta: 0.0                                # > 0 requires models.ref_model
      per_prompt_stat_tracking: true
      global_std: false
      skip_timesteps: 1                        # trailing rollout steps exempt from loss
      random_policy_timestep: true             # one random step per group vs full grid
      policy_loss_scaling: 100.0
      policy_timestep_reweight: false
      policy_timestep_weight_eps: 1.0e-6

LR schedules are NOT engine config: an ``lr_scheduler`` node sits inside a
model's optimizer item -- see common/lr_scheduler. They step once per
optimizer step (``scheduler.step(iter)`` before that step's ``optimize()``).
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections import Counter
from typing import Any

import torch

from common.data import build_dataloader
from common.distributed import ops
from common.engine import BaseEngine
from common.logging import get_logger
from common.lr_scheduler import build_lr_schedulers
from common.meter import AverageMeter, get_running_accumulator, get_running_average_meter
from common.model import build_models
from common.optim import build_optimizers, checkpoint_models
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed

logger = get_logger()


class PerPromptStatTracker:
    """Per-prompt reward mean/std tracking across all ranks.

    RAVEN: group_relative_policy_optimization.py:30-148 (PerPromptStatTracker).
    Two mechanical adaptations:
      * ``project.utils.comm`` -> ``common.distributed.ops``; the
        common ``all_gather`` is SYNCHRONOUS and returns the gathered list
        directly, so the async handles are always ``None`` (the finish side
        already treats ``None`` as "nothing to wait on").
      * ``project.utils.running`` meter -> ``common.meter``: no ``cnt`` kwarg,
        per-observation averaging (metrics only).
    """

    def __init__(self, global_std: bool = False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self._pending = []

    def _prompt_to_id(self, prompt: str) -> int:
        # RAVEN: grpo.py:37-42
        return int.from_bytes(
            hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
            signed=True,
        )

    def _update(self, prompt_ids, rewards):
        # RAVEN: grpo.py:44-63
        rewards = rewards.detach().cpu().to(dtype=torch.float32)
        unique_prompts = list(dict.fromkeys(prompt_ids))
        advantages = torch.empty_like(rewards, dtype=torch.float32)

        for prompt in unique_prompts:
            prompt_indices = [idx for idx, value in enumerate(prompt_ids) if value == prompt]
            prompt_rewards = rewards[prompt_indices]
            self.stats.setdefault(prompt, []).append(prompt_rewards)
            self.history_prompts.add(prompt)

        global_std = rewards.std(unbiased=False).clamp_min(1e-4) if self.global_std else None
        for prompt in unique_prompts:
            prompt_indices = [idx for idx, value in enumerate(prompt_ids) if value == prompt]
            prompt_rewards = rewards[prompt_indices]
            history = torch.cat(self.stats[prompt], dim=0)
            mean = history.mean()
            std = global_std if global_std is not None else history.std(unbiased=False).clamp_min(1e-4)
            advantages[prompt_indices] = (prompt_rewards - mean) / (std + 1e-5)
        return advantages

    def start_update(self, prompts, rewards):
        # RAVEN: grpo.py:65-80 — >>>>>>>>>>>>>> RAVEN adaptation: sync all_gather
        # NOTE: upstream issues the two all_gathers with async_op=True and stashes
        # the handles (RAVEN: grpo.py:75-79):
        #         id_handle = comm.all_gather(prompt_ids, gathered_ids, async_op=True)
        #         reward_handle = comm.all_gather(flat_rewards, gathered_rewards, async_op=True)
        # common.distributed.ops.all_gather is synchronous and returns the list, so
        # the gather completes HERE and both handles are None. Numerics unchanged;
        # only the compute/comm overlap is dropped.
        # <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
        repeated_prompts = []
        for _ in range(int(rewards.shape[0])):
            repeated_prompts.extend(prompts)
        prompt_ids = torch.tensor(
            [self._prompt_to_id(prompt) for prompt in repeated_prompts],
            dtype=torch.int64,
            device=rewards.device,
        )
        flat_rewards = rewards.reshape(-1)
        gathered_ids = ops.all_gather(prompt_ids)
        gathered_rewards = ops.all_gather(flat_rewards)
        id_handle = None
        reward_handle = None
        self._pending.append((prompt_ids, gathered_ids, gathered_rewards, id_handle, reward_handle, rewards.shape, rewards.device))

    def finish_updates(self):
        # RAVEN: grpo.py:82-145
        local_rank = ops.get_rank()
        local_prompt_ids = []
        local_shapes = []
        local_devices = []
        synced_prompt_ids = []
        synced_rewards = []

        for prompt_ids, gathered_ids, gathered_rewards, id_handle, reward_handle, reward_shape, reward_device in self._pending:
            if id_handle is not None:
                id_handle.wait()
            if reward_handle is not None:
                reward_handle.wait()

            local_prompt_ids.append(prompt_ids)
            local_shapes.append(reward_shape)
            local_devices.append(reward_device)
            synced_prompt_ids.append(torch.cat(gathered_ids, dim=0))
            synced_rewards.append(torch.cat(gathered_rewards, dim=0))

        all_synced_prompt_ids = torch.cat(synced_prompt_ids, dim=0).cpu().tolist()
        all_synced_rewards = torch.cat(synced_rewards, dim=0)
        local_all_prompt_ids = torch.cat(local_prompt_ids, dim=0).cpu().tolist()
        global_prompt_counts = Counter(all_synced_prompt_ids)
        local_prompt_counts = Counter(local_all_prompt_ids)
        if local_prompt_counts:
            local_group_sizes = torch.tensor(
                [local_prompt_counts[prompt] for prompt in local_prompt_counts],
                dtype=torch.float32,
            )
            effective_group_sizes = torch.tensor(
                [global_prompt_counts[prompt] for prompt in local_prompt_counts],
                dtype=torch.float32,
            )
            effective_sync_group_sizes = effective_group_sizes / local_group_sizes.clamp_min(1.0)
            meter = get_running_average_meter()
            meter.put_scalar("running/reward_local_group_size", local_group_sizes.mean())
            meter.put_scalar("running/reward_effective_group_size", effective_group_sizes.mean())
            meter.put_scalar("running/reward_effective_sync_group_size", effective_sync_group_sizes.mean())
        all_synced_advantages = self._update(all_synced_prompt_ids, all_synced_rewards)

        local_advantages = []
        global_offset = 0
        for prompt_ids, reward_shape, reward_device in zip(local_prompt_ids, local_shapes, local_devices):
            local_offset = global_offset + local_rank * prompt_ids.numel()
            local_advantages.append(
                all_synced_advantages[local_offset:local_offset + prompt_ids.numel()].to(
                    device=reward_device, dtype=torch.float32
                ).reshape(reward_shape)
            )
            global_offset += ops.get_world_size() * prompt_ids.numel()

        local_advantages = torch.cat(local_advantages, dim=0)
        self._pending.clear()
        self.clear()
        return local_advantages

    def clear(self):
        self.stats = {}


def select_reward_scores(tensor_rewards, reward_field, reward_aggregation, groups_per_infer, batch_size):
    """Pick/combine reward fields from the reward dict.

    RAVEN: group_relative_policy_optimization.py:151-188 (select_reward_scores),
    ported with its config-validation asserts dropped (native KeyError /
    ZeroDivisionError are loud enough).
    """
    if reward_field is None:
        return next(iter(tensor_rewards.values()))
    if isinstance(reward_field, str):
        return tensor_rewards[reward_field]

    if isinstance(reward_field, dict):
        reward_items = [(field, float(weight)) for field, weight in reward_field.items()]
    else:
        reward_items = [(field, 1.0) for field in reward_field]
    normalizer = sum(abs(weight) for _, weight in reward_items)

    if reward_aggregation == "sum_then_normalize":
        return torch.stack([
            tensor_rewards[field].to(dtype=torch.float32) * weight
            for field, weight in reward_items
        ], dim=0).sum(dim=0) / normalizer
    if reward_aggregation == "normalize_then_sum":
        rewards = torch.stack([
            tensor_rewards[field].to(dtype=torch.float32)
            for field, _ in reward_items
        ], dim=0).reshape(len(reward_items), groups_per_infer, batch_size)
        weights = torch.tensor(
            [weight for _, weight in reward_items],
            device=rewards.device,
            dtype=rewards.dtype,
        ).view(-1, 1, 1)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        return (((rewards - mean) / (std + 1e-5)) * weights).sum(dim=0).reshape(-1) / normalizer
    raise ValueError(f"Unsupported reward_aggregation: {reward_aggregation}")


def _deepcopy_with_tensor(value):
    """RAVEN: project/utils/misc.py (deepcopy_with_tensor) — tensors are cloned,
    containers recursed, everything else deep-copied."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_deepcopy_with_tensor(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deepcopy_with_tensor(v) for v in value)
    if isinstance(value, dict):
        return {k: _deepcopy_with_tensor(v) for k, v in value.items()}
    return copy.deepcopy(value)


class GroupRelativePolicyOptimization(BaseEngine):
    """GRPO engine: no-grad group rollout + sliced policy optimization phase."""

    def __init__(self, config: Any, meta_model: Any):
        self._start_time = time.monotonic()
        self.config = config
        self.engine_config = config.engine
        if "early_stop_hours" in self.engine_config:
            self._early_stop_seconds = float(self.engine_config.early_stop_hours) * 60 * 60
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
        if cfg.get("reverse_advantage", False):
            # RAVEN: grpo.py:257-258 — reverse_advantage forces the policy sign flip
            cfg["reverse_policy_advantage"] = True
        self.policy_step_models = list(cfg.get("policy_step_models", ["backbone"]))
        self._val_interval = int(cfg.get("val_interval", 0))
        self._val_before_train = cfg.get("val_before_train", False)

        # -- build: models -> optimizers -> dataloader ----------------------------
        # Order matters: optimizers hold post-placement (sharded) parameters.
        self.models = build_models(config.models)
        self.optimizers = build_optimizers(self.models, config.models)
        self.lr_schedulers = build_lr_schedulers(self.optimizers, config.models)
        self.dataloader = build_dataloader(config.data, seed=int(cfg.seed))

        # RAVEN: grpo.py:273-274
        self.per_prompt_stat_tracker = PerPromptStatTracker(global_std=bool(cfg.get("global_std", False))) \
            if cfg.get("per_prompt_stat_tracking", True) else None

        # -- checkpoint state tree + resume ---------------------------------------
        # Everything that must survive a restart, and nothing else. ctx never
        # appears here by design. The stat tracker is stateless across save
        # boundaries in RAVEN too (stats are cleared every finish_updates).
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
            offline = int(cfg.offline)
            ga = int(cfg.ga_steps)
            assert self.iter == self.resumed_step * ga, (
                f"checkpoint step {self.resumed_step} with ga_steps={ga} requires runtime iter "
                f"{self.resumed_step * ga}, got {self.iter}"
            )
            assert self.iter % (offline * ga) == 0, (
                f"resumed iter={self.iter} is not aligned to the grid boundary "
                f"offline*ga={offline * ga} (checkpoints only exist at grid boundaries)"
            )
            assert self.dataloader.consumed == self.iter // (offline * ga), (
                f"resume dataloader mismatch: consumed={self.dataloader.consumed} "
                f"iter//(offline*ga)={self.iter // (offline * ga)}"
            )
        if not self._resumed:
            self.sync_ema_from_model()  # step-0 contract: EMA starts as an exact copy

    # -------------------------------------------------------------- the run --
    def run(self) -> None:
        # UP guard: GRPO rollout/policy phases are not SP-aware yet -- silent
        # per-rank full-sequence duplication would corrupt advantages/gradients.
        assert self._train_up_size == 1 and self._validation_up_size is None, (
            "GRPO does not support up_size > 1 yet: rollout/policy phases are "
            "not SP-aware (causal_wan_t2v_grpo has no unified-parallel support)"
        )
        cfg = self.engine_config
        offline = int(cfg.offline)
        ga = int(cfg.ga_steps)
        save_interval = int(cfg.save_interval)
        val_interval = int(self._val_interval)
        total_iters = int(cfg.training_steps) * ga  # backwards (iter's unit)

        assert int(cfg.training_steps) % offline == 0, (
            f"training_steps={int(cfg.training_steps)} must be divisible by offline={offline}"
        )
        assert save_interval % offline == 0, (
            f"save_interval={save_interval} must be divisible by offline={offline}"
        )
        assert val_interval == 0 or val_interval % offline == 0, (
            f"val_interval={val_interval} must be 0 or divisible by offline={offline}"
        )
        assert self.iter % (offline * ga) == 0, (
            f"iter={self.iter} is not aligned to the grid boundary offline*ga={offline * ga}"
        )

        logger.info(
            "Training: backwards %d -> %d (optimizer steps=%d, group_size=%d, "
            "groups_per_infer=%d, offline=%d, ga_steps=%d; total rollouts=%d)",
            self.iter, total_iters, int(cfg.training_steps), int(cfg.group_size),
            int(cfg.groups_per_infer), offline, ga, int(cfg.training_steps) // offline,
        )

        if cfg.get("save_before_train", False) and not self._resumed:
            self._runtime_state["iter"] = self.iter
            self.config.checkpointer.save(self.state_tree, step=self.iter // ga)

        if self._val_before_train and not self._resumed:
            self.validate(step=0)

        step_timer = time.perf_counter()
        meter = AverageMeter()
        rank = ops.get_rank()
        while self.iter < total_iters:
            # -- one group rollout amortized over `offline` optimizer steps ------
            rollout_idx = self.iter // (offline * ga)
            tick = time.perf_counter()
            batch = next(self.dataloader)
            meter.put_scalar("timer/dataload", time.perf_counter() - tick)

            ctx: dict[str, Any] = {
                "config": self.config,
                "iter": rollout_idx,
                "batch": batch,
                "models": self.models,
                "rng": RandomState(seed=combine_seed(int(cfg.seed), "prepare", rollout_idx, rank)),
            }
            tick = time.perf_counter()
            ctx = self.meta.prepare_inputs(ctx)
            meter.put_scalar("timer/prepare", time.perf_counter() - tick)
            # RAVEN: grpo.py:386-389 (data stats, logged right after prepare)
            get_running_accumulator().update({
                "data/num_total_tokens": sum(ctx["inputs"].seqlens),
                "data/num_samples": ctx["inputs"].batch_size,
            })

            # -- rollout phase (no-grad) ------------------------------------------
            # RAVEN: grpo.py:392-394 — eval mode for rollout, train mode for policy
            backbone = self.models["backbone"]
            backbone.train(False)
            ctx["rng"] = RandomState(seed=combine_seed(int(cfg.seed), "rollout", rollout_idx, rank))
            tick = time.perf_counter()
            ctx = self.meta.rollout(ctx)
            meter.put_scalar("timer/rollout", time.perf_counter() - tick)
            backbone.train(True)

            # -- engine bookkeeping: rewards -> advantages -> groups -> slices -----
            advantages = self._prepare_advantages(ctx)
            group_rollouts = self._split_group_rollouts(ctx["infer_rollouts"])
            del ctx
            policy_rng = RandomState(seed=combine_seed(int(cfg.seed), "policy", rollout_idx, rank))
            policy_slice_indices = self._build_policy_slice_indices(group_rollouts, advantages, policy_rng)
            chunk_size = len(policy_slice_indices) // offline

            # -- `offline` optimizer steps over the shuffled grid -----------------
            for chunk_idx in range(offline):
                chunk = policy_slice_indices[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                # LR advances once per optimizer step with its 0-based index
                # (RAVEN: diffusion_finetuning.py:154-155).
                logical_step = self.iter // ga
                for scheduler in self.lr_schedulers.values():
                    scheduler.step(logical_step)

                tick = time.perf_counter()
                meter.update(self._policy_phase(group_rollouts, advantages, chunk))
                meter.put_scalar("timer/policy", time.perf_counter() - tick)

                # iter now points at this optimizer step's LAST backward: common's
                # ga-aware mechanics (at_interval/save/log_metrics) apply verbatim.
                self.iter += ga - 1

                # One clean node per rollout grid: the grid boundary is where
                # log/save/val/runtime-stop all fire -- resume then needs zero
                # policy-phase state (dataloader.consumed == iter // (offline*ga)).
                if (self.iter + 1) % (offline * ga) == 0:
                    step_timer, stop = self.process_due(
                        meter, step_timer,
                        total_iters=total_iters,
                    )
                    if stop:
                        return

                self.iter += 1

            del group_rollouts, advantages, policy_slice_indices

        logger.info("Training complete at iter=%d", self.iter)

    # -------------------------------------------------------- rollout follow --
    def _prepare_advantages(self, ctx: dict[str, Any]) -> torch.Tensor:
        """Select reward scores per infer, meter them, normalize into advantages.

        RAVEN: grpo.py:339-414 — the selection/tracker calls RAVEN makes inside
        its no-grad rollout loop run here instead (identical order and values;
        reward outputs are already detached). The upstream meter calls carry a
        ``cnt`` kwarg that common.meter does not support -- dropped (metrics only).
        """
        cfg = self.engine_config
        pos_inputs = ctx["inputs"]
        groups_per_infer = int(cfg.groups_per_infer)
        batch_size = int(pos_inputs.batch_size.item()) \
            if isinstance(pos_inputs.batch_size, torch.Tensor) else int(pos_inputs.batch_size)

        rewards_per_infer = []
        for infer in ctx["infer_rollouts"]:
            reward_output = infer["tensor_rewards"]
            if isinstance(reward_output, torch.Tensor):
                reward_scores = reward_output
            else:
                tensor_rewards = {
                    key: value
                    for key, value in reward_output.items()
                    if isinstance(value, torch.Tensor)
                }
                for key, value in tensor_rewards.items():
                    if value.numel() == 0:
                        continue
                    get_running_average_meter().put_scalar(
                        f"rewards/{key}",
                        value.detach().to(dtype=torch.float32).mean(),
                    )
                reward_scores = select_reward_scores(
                    tensor_rewards,
                    cfg.reward_field,
                    getattr(cfg, "reward_aggregation", "sum_then_normalize"),
                    groups_per_infer,
                    batch_size,
                )
            infer_rewards = reward_scores.to(device=ops.get_device(), dtype=torch.float32).reshape(
                groups_per_infer, batch_size
            )
            if self.per_prompt_stat_tracker is not None:
                self.per_prompt_stat_tracker.start_update(pos_inputs.prompts, infer_rewards)
            rewards_per_infer.append(infer_rewards)

        rewards = torch.cat(rewards_per_infer, dim=0)
        reward_std = rewards.std(dim=0, keepdim=True, unbiased=False)
        reward_max = rewards.max(dim=0).values
        reward_min = rewards.min(dim=0).values
        if self.per_prompt_stat_tracker is None:
            reward_mean = rewards.mean(dim=0, keepdim=True)
            advantages = (rewards - reward_mean) / (reward_std + 1e-5)
        else:
            advantages = self.per_prompt_stat_tracker.finish_updates()

        get_running_average_meter().update({
            "running/reward_mean": rewards.mean(),
            "running/reward_std": reward_std.mean(),
            "running/reward_group_max_mean": reward_max.mean(),
            "running/reward_group_min_mean": reward_min.mean(),
            "running/reward_gap": (reward_max - reward_min).mean(),
            "running/reward_advantage_abs": advantages.abs().mean(),
        })
        return advantages

    def _split_group_rollouts(self, infer_rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split each infer batch into ``groups_per_infer`` per-prompt groups.

        RAVEN: grpo.py:416-439. The trajectory parts go through
        ``_deepcopy_with_tensor`` exactly like RAVEN's wrap; masked tensors from
        boolean indexing are fresh tensors already, the copy keeps list elements
        decoupled from the (soon deleted) infer rollouts.
        """
        cfg = self.engine_config
        groups_per_infer = int(cfg.groups_per_infer)
        group_rollouts = []
        for infer in infer_rollouts:
            infer_batch_size = int(infer["pos"].batch_size.item()) \
                if isinstance(infer["pos"].batch_size, torch.Tensor) else int(infer["pos"].batch_size)
            assert infer_batch_size % groups_per_infer == 0, \
                f"infer batch size {infer_batch_size} must be divisible by groups_per_infer {groups_per_infer}"
            batch_size = infer_batch_size // groups_per_infer
            device = infer["pos"].batch_size.device \
                if isinstance(infer["pos"].batch_size, torch.Tensor) else ops.get_device()
            for group_idx in range(groups_per_infer):
                group_start = group_idx * batch_size
                group_end = group_start + batch_size
                group_mask = torch.zeros(infer_batch_size, device=device, dtype=torch.bool)
                group_mask[group_start:group_end] = True
                group_rollouts.append({
                    "pos": self.meta.mask_inputs(infer["pos"], group_mask),
                    "neg": self.meta.mask_inputs(infer["neg"], group_mask) if infer["neg"] is not None else None,
                    **_deepcopy_with_tensor({
                        "x0s": self.meta.mask_tensors(infer["x0s"], group_mask),
                        "trajectory_xt": [self.meta.mask_tensors(xt, group_mask) for xt in infer["trajectory_xt"]],
                        "rollout_t": [self.meta.mask_tensors(t, group_mask) for t in infer["rollout_t"]],
                        "rollout_s": [self.meta.mask_tensors(s, group_mask) for s in infer["rollout_s"]],
                    }),
                })
        return group_rollouts

    def _build_policy_slice_indices(
        self,
        group_rollouts: list[dict[str, Any]],
        advantages: torch.Tensor,
        rng: RandomState,
    ) -> list[tuple[int, int]]:
        """Build the shuffled (group_idx, step_idx) policy slice grid.

        RAVEN: grpo.py:440-465 — one random step per group when
        ``random_policy_timestep`` else the full cartesian grid; always shuffled
        with the phase rng (CPU generator off-device, CUDA generator on-device).
        """
        cfg = self.engine_config
        num_policy_steps = len(group_rollouts[0]["rollout_t"]) - int(cfg.skip_timesteps)
        generator = rng.torch_cuda_generator if advantages.is_cuda else rng.torch_generator
        if getattr(cfg, "random_policy_timestep", False):
            random_step_indices = torch.randint(
                num_policy_steps,
                (len(group_rollouts),),
                device=advantages.device,
                generator=generator,
            )
            policy_slice_indices = [
                (group_idx, step_idx)
                for group_idx, step_idx in enumerate(random_step_indices.cpu().tolist())
            ]
        else:
            policy_slice_indices = [
                (group_idx, step_idx)
                for group_idx in range(len(group_rollouts))
                for step_idx in range(num_policy_steps)
            ]
        # divisibility contract (see module docstring):
        #   offline | (group_size x timesteps)  and  ga_steps | (group_size x timesteps // offline)
        n = len(policy_slice_indices)
        assert n % int(cfg.offline) == 0, (
            f"offline ({int(cfg.offline)}) must divide the policy grid "
            f"({n} = group_size x timesteps slices)"
        )
        assert (n // int(cfg.offline)) % int(cfg.ga_steps) == 0, (
            f"ga_steps ({int(cfg.ga_steps)}) must divide the per-offline chunk "
            f"({n // int(cfg.offline)} = grid/offline slices)"
        )
        if len(policy_slice_indices) > 1:
            permutation = torch.randperm(len(policy_slice_indices), device=advantages.device, generator=generator)
            policy_slice_indices = [policy_slice_indices[int(idx)] for idx in permutation.cpu().tolist()]
        return policy_slice_indices

    # ------------------------------------------------------------ policy phase --
    def _policy_phase(
        self,
        group_rollouts: list[dict[str, Any]],
        advantages: torch.Tensor,
        policy_slice_indices: list[tuple[int, int]],
    ) -> dict[str, float]:
        """ONE optimizer step over a shuffled chunk of the grid: ``ga_steps``
        backwards of ``len(chunk)/ga_steps`` merged slices, then optimize.

        RAVEN: grpo.py:467-507 (per-chunk body). Every backward is scaled
        ``1/ga_steps`` == RAVEN's ``len(sub)/len(chunk)`` exactly; FSDP2
        gradient sync opens only for the LAST backward of the step
        (``set_gradient_sync``). ``self.iter`` is owned by ``run()`` (it counts
        backwards); LR stepping happens there before this phase is called.
        """
        cfg = self.engine_config
        ga = int(cfg.ga_steps)
        sub_size = len(policy_slice_indices) // ga

        sub_losses = []
        for backward_idx in range(ga):
            sub = policy_slice_indices[backward_idx * sub_size:(backward_idx + 1) * sub_size]
            self.set_gradient_sync(backward_idx == ga - 1)
            ctx: dict[str, Any] = {
                "config": self.config,
                "models": self.models,
                "group_rollouts": group_rollouts,
                "advantages": advantages,
                "policy_slice_indices": sub,
            }
            ctx = self.meta.prepare_policy(ctx)
            ctx = self.meta.policy_forward(ctx)
            ctx = self.meta.policy_loss(ctx)
            loss = ctx["policy_loss"]
            # RAVEN: grpo.py:492 — len(sub)/len(chunk) == 1/ga_steps exactly
            self.backward(loss / ga)
            sub_losses.append(loss.detach())
            del ctx, loss

        norm_dict = self.optimize(names=list(self.policy_step_models))

        # RAVEN weights sub-losses by slice count; equal sub sizes (ga | chunk)
        # make this the plain mean.
        log_dict = {"train/loss": torch.stack(sub_losses).mean().item()}
        log_dict.update(norm_dict)
        return log_dict
