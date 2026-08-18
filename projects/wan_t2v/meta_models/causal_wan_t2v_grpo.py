"""CausalWanT2VGRPO: all GRPO algorithm primitives for causal Wan2.1 T2V.

One project == one engine, so this class has NO genericity: it inherits only the
causal-Wan machinery it shares byte-for-byte with ``CausalWanT2VDMD``
(``CausalWanT2V``); the public surface is exactly the five training ctx
primitives plus validation; everything else is a private recipe.

CM-GRPO (consistency-model group relative policy optimization): a no-grad group
rollout produces ``group_size`` trajectories per prompt batch together with their
reward tensors; the engine turns rewards into advantages and slices the
(group, timestep) grid; each policy slice replays one Gaussian transition step
through the backbone and regresses the predicted x0 toward an implicit
score-grad target (advantage-weighted score matching on the consistency
kernel), with an optional KL anchor to a reference model.

ctx contract::

    prepare_inputs   reads batch/models/rng/config      writes inputs, neg_inputs
    rollout          reads models/inputs/neg_inputs/rng writes infer_rollouts          (no-grad)
    prepare_policy   reads group_rollouts/advantages/policy_slice_indices
                                                        writes pos_step_inputs, neg_step_inputs,
                                                        next_xt, step_s, slice_advantages, slice_step_indices
    policy_forward   reads models/pos_step_inputs/neg_step_inputs
                                                        writes policy_pred (, ref_pred)  (graph attached)
    policy_loss      reads pos_step_inputs/policy_pred/step_s/slice_advantages/...
                                                        writes policy_loss

``infer_rollouts`` is a list of ``groups_per_infer``-merged rollout dicts (keys:
``pos``, ``neg``, ``x0s``, ``trajectory_xt``, ``rollout_t``, ``rollout_s``,
``tensor_rewards``); the engine owns reward selection, advantage bookkeeping and
the group split (``mask_inputs``/``mask_tensors`` below are its helpers).

Expected configuration shape::

    meta_model:
      dummy_latents: true
      guidance_min: 3.0
      guidance_max: 3.0
      default_neg_prompt: ""
      z_dim: 16
      chunk_wise_weighting:
        policy_losses: -1.0
        kl_losses: -1.0

Note: ``config.diffusion`` provides 3 components: sampling_timesteps, schedule,
sampler. GRPO knobs (group_size, beta, adv_clip_max, ...) live in
``config.engine`` and are read from ctx, not at construction time.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule

from common.distributed.ops import get_device
from common.logging import get_logger
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, execution_phase

from modeling.reward_models import BaseRewardModel
from .causal_wan_t2v import CausalWanT2V, ForwardInput

logger = get_logger()


class CausalWanT2VGRPO(CausalWanT2V):
    """See module docstring for the ctx contract."""

    def __init__(self, config):
        super().__init__(config)
        mm = config.meta_model
        # RAVEN: causal_wan2_1_t2v.py:1087 — reward_fn decodes latents only when
        # channels == z_dim; kept as an explicit config knob (RAVEN reads it from
        # its meta_model_config).
        self.z_dim = int(mm.z_dim)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, neg_inputs, rng. Writes: infer_rollouts. (no-grad)

        RAVEN: group_relative_policy_optimization.py:288-378 (_build_group_rollouts).
        One infer merges ``groups_per_infer`` copies of the prompt batch, draws fresh
        sampling noises, runs one chunked AR rollout with trajectory capture, and
        scores the decoded latents with the reward model(s). Reward SELECTION and
        advantage bookkeeping stay on the engine (the stat tracker is engine state).
        """
        models = ctx["models"]
        rng = ctx["rng"]
        backbone = models["backbone"]
        device = get_device()

        engine_cfg = ctx["config"].engine
        groups_per_infer = int(engine_cfg.groups_per_infer)
        assert int(engine_cfg.group_size) % groups_per_infer == 0, \
            f"group_size {int(engine_cfg.group_size)} must be divisible by groups_per_infer {groups_per_infer}"
        num_infers = int(engine_cfg.group_size) // groups_per_infer

        pos_inputs = ctx["inputs"]
        neg_inputs = ctx["neg_inputs"]
        # The neg chain has consumers only when the ref path is armed (beta > 0,
        # external-CFG ref model) or the policy itself is a legacy external-CFG
        # model (guidance_embeds None). Otherwise every merge/clone/mask/repack
        # of neg inputs is dead bookkeeping — skip constructing it entirely.
        use_neg = float(getattr(engine_cfg, "beta", 0.0)) > 0.0 or backbone.guidance_embeds is None

        # RAVEN: grpo.py:299-303
        self.sampling_timesteps.set_timesteps(seqlen=pos_inputs.seqlens, device=device)
        assert self.sampling_timesteps.timesteps.ndim == 1, \
            "GroupRelativePolicyOptimization currently expects 1D sampling timesteps"
        batch_size = int(pos_inputs.batch_size.item()) \
            if isinstance(pos_inputs.batch_size, torch.Tensor) else int(pos_inputs.batch_size)
        del batch_size  # only the engine's reward reshape needs it

        infer_rollouts = []
        for _ in range(num_infers):
            # RAVEN: grpo.py:304-311 — duplicate the prompt batch groups_per_infer
            # times. merge_inputs(detach=True) returns tensor-level copies already,
            # and the rollout clones below mirror RAVEN's deepcopy_with_tensor.
            infer_pos_inputs = self.merge_inputs(*[pos_inputs] * groups_per_infer)
            infer_neg_inputs = self.merge_inputs(*[neg_inputs] * groups_per_infer) if use_neg else None

            rollout_pos_inputs = self._clone_inputs(infer_pos_inputs)
            rollout_neg_inputs = self._clone_inputs(infer_neg_inputs) if use_neg else None

            # RAVEN: grpo.py:313-316 — sample_noises (base_meta_model.py:315-324),
            # drawn BEFORE infer; SAME noises on neg (RAVEN engine semantics).
            self.sampling_timesteps.set_timesteps(seqlen=rollout_pos_inputs.seqlens, device=device)
            sampling_noises = [
                torch.empty_like(latent).normal_(generator=rng.torch_cuda_generator)
                for latent in rollout_pos_inputs.latents
            ]
            rollout_pos_inputs.noises = sampling_noises
            if use_neg:
                rollout_neg_inputs.noises = sampling_noises

            # RAVEN: grpo.py:318-321 — per-step (t, s) pairs expanded to the batch.
            slice_batch_size = int(rollout_pos_inputs.batch_size.item()) \
                if isinstance(rollout_pos_inputs.batch_size, torch.Tensor) else int(rollout_pos_inputs.batch_size)
            rollout_t = [t.expand(slice_batch_size) for t in self.sampling_timesteps.timesteps]
            rollout_s = [self.sampling_timesteps.get_next_timesteps(t) for t in rollout_t]

            # RAVEN: grpo.py:323-333 (infer with return_trajectory=True)
            latent_x0s, trajectory_xt, trajectory_pred = self._infer(
                model=backbone,
                rng=rng,
                pos_inputs=rollout_pos_inputs,
                neg_inputs=rollout_neg_inputs,
            )
            # trajectory_pred is produced by the transcribed infer but intentionally
            # NOT kept: nothing downstream reads it (RAVEN discards it as `_`).
            del trajectory_pred

            # RAVEN: grpo.py:334-338 (reward_fn; selection/metering is engine-side)
            tensor_rewards = self._reward_fn(
                models=models,
                inputs=infer_pos_inputs,
                tensors=latent_x0s,
            )

            infer_rollouts.append({
                "pos": infer_pos_inputs,
                "neg": infer_neg_inputs,
                "x0s": latent_x0s,
                "trajectory_xt": trajectory_xt,
                "rollout_t": rollout_t,
                "rollout_s": rollout_s,
                "tensor_rewards": tensor_rewards,
            })

        ctx["infer_rollouts"] = infer_rollouts
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_policy(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: group_rollouts, advantages, policy_slice_indices.
        Writes: pos_step_inputs, neg_step_inputs, next_xt, step_s,
        slice_advantages, slice_step_indices, num_policy_steps."""
        # RAVEN: group_relative_policy_optimization.py:515-548 (policy_training_step
        # assembly): per (group, step) slice rebuild inputs carrying that step's
        # latents/timestep/noisy xt, then merge across slices.
        group_rollouts = ctx["group_rollouts"]
        advantages = ctx["advantages"]
        policy_slice_indices = ctx["policy_slice_indices"]
        engine_cfg = ctx["config"].engine
        num_policy_steps = len(group_rollouts[0]["rollout_t"]) - int(engine_cfg.skip_timesteps)

        # Data-driven mirror of the rollout-side neg gate: when the rollout
        # skipped the neg chain, group["neg"] is None and the per-slice neg
        # clone/repack below is dead work with no consumer.
        use_neg = group_rollouts[0]["neg"] is not None

        pos_step_inputs = []
        neg_step_inputs = []
        next_xts = []
        step_ss = []
        step_advantages = []
        step_indices = []
        for group_idx, step_idx in policy_slice_indices:
            group = group_rollouts[group_idx]
            latent_x0s = group["x0s"]
            trajectory_xt = group["trajectory_xt"]
            trajectory_t = group["rollout_t"]
            trajectory_s = group["rollout_s"]
            pos_inputs_i = self._clone_inputs(group["pos"])
            pos_inputs_i.latents = latent_x0s
            pos_inputs_i.timesteps = trajectory_t[step_idx]
            pos_inputs_i.xts = trajectory_xt[step_idx]
            pos_step_inputs.append(pos_inputs_i)
            if use_neg:
                neg_inputs_i = self._clone_inputs(group["neg"])
                neg_inputs_i.latents = latent_x0s
                neg_inputs_i.timesteps = trajectory_t[step_idx]
                neg_inputs_i.xts = trajectory_xt[step_idx]
                neg_step_inputs.append(neg_inputs_i)
            next_xts.append(trajectory_xt[step_idx + 1] if step_idx + 1 < len(trajectory_xt) else latent_x0s)
            step_ss.append(trajectory_s[step_idx])
            group_advantages = advantages[group_idx].reshape(-1)
            step_advantages.append(group_advantages)
            step_indices.extend([step_idx] * int(group_advantages.numel()))

        pos_step_inputs = self.merge_inputs(*pos_step_inputs)
        neg_step_inputs = self.merge_inputs(*neg_step_inputs) if use_neg else None
        next_xt = self.merge_tensors(next_xts)
        s = torch.cat(step_ss, dim=0)
        slice_advantages = torch.cat(step_advantages, dim=0)
        # RAVEN: grpo.py:547 — reverse_policy_advantage flips the advantage sign.
        if bool(getattr(engine_cfg, "reverse_policy_advantage", False)):
            slice_advantages = -slice_advantages

        ctx["pos_step_inputs"] = pos_step_inputs
        ctx["neg_step_inputs"] = neg_step_inputs
        ctx["next_xt"] = next_xt
        ctx["step_s"] = s
        ctx["slice_advantages"] = slice_advantages
        ctx["slice_step_indices"] = step_indices
        ctx["num_policy_steps"] = num_policy_steps
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def policy_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, pos_step_inputs, neg_step_inputs.
        Writes: policy_pred (, ref_pred when engine.beta > 0). (graph attached)"""
        # RAVEN: group_relative_policy_optimization.py:550-556 + 635-640
        models = ctx["models"]
        backbone = models["backbone"]
        pos_step_inputs = ctx["pos_step_inputs"]
        neg_step_inputs = ctx["neg_step_inputs"]

        beta = float(getattr(ctx["config"].engine, "beta", 0.0))
        # RAVEN: grpo.py:550-552 — _block_mask MUTATES sample_lens/split_lens/
        # attn_modes with padding entries, so the ref pass needs pristine clones.
        if beta > 0.0:
            ref_pos_step_inputs = self._clone_inputs(pos_step_inputs)
            ref_neg_step_inputs = self._clone_inputs(neg_step_inputs)

        if isinstance(backbone, FSDPModule):
            backbone.unshard()
        ctx["policy_pred"] = self._pred(backbone, pos_step_inputs, neg_step_inputs)

        if beta > 0.0:
            ref_model = models["ref_model"]
            with torch.no_grad():
                if isinstance(ref_model, FSDPModule):
                    ref_model.unshard()
                # RAVEN: causal_wan2_1_t2v.py:886-891 — pred_cfg dispatches a
                # non-causal WanModel (this trial's ref_model) to the flat path.
                ctx["ref_pred"] = self._pred_cfg_flat(ref_model, ref_pos_step_inputs, ref_neg_step_inputs)
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def policy_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: pos_step_inputs, policy_pred, next_xt, step_s, slice_advantages,
        slice_step_indices, num_policy_steps (, ref_pred). Writes: policy_loss."""
        # RAVEN: group_relative_policy_optimization.py:557-664 — REINFORCE on the
        # Gaussian transition kernel realized as an x0-space regression toward an
        # implicit score-grad target; optional KL anchor to the reference model.
        engine_cfg = ctx["config"].engine
        pos_step_inputs = ctx["pos_step_inputs"]
        pred = ctx["policy_pred"]
        next_xt = ctx["next_xt"]
        s = ctx["step_s"]
        slice_advantages = ctx["slice_advantages"]
        slice_step_indices = ctx["slice_step_indices"]
        num_policy_steps = ctx["num_policy_steps"]

        # RAVEN: grpo.py:557 (transition_kernel — base_meta_model.py:368-381)
        mean, std = self.sampler.transition_kernel(pred, pos_step_inputs.xts, pos_step_inputs.timesteps, s)

        # RAVEN: grpo.py:558-577 (policy_timestep_reweight branch)
        if bool(getattr(engine_cfg, "policy_timestep_reweight", False)):
            if isinstance(std, list):
                slice_timestep_weights = torch.stack([
                    std_i.detach().to(device=pos_step_inputs.timesteps.device, dtype=torch.float32).mean()
                    for std_i in std
                ])
            else:
                slice_timestep_weights = std.detach().to(device=pos_step_inputs.timesteps.device, dtype=torch.float32)
                slice_timestep_weights = slice_timestep_weights.reshape(slice_timestep_weights.shape[0], -1).mean(dim=1)
            dt = (
                (pos_step_inputs.timesteps.to(dtype=torch.float32) - s.to(dtype=torch.float32)).abs()
                / self.sampler._sigma_denominator()
            ).clamp_min(float(getattr(engine_cfg, "policy_timestep_weight_eps", 1e-6)))
            slice_timestep_weights = slice_timestep_weights / dt
        else:
            slice_timestep_weights = torch.ones_like(
                pos_step_inputs.timesteps,
                device=pos_step_inputs.timesteps.device,
                dtype=torch.float32,
            )

        # RAVEN: grpo.py:579-585
        adv_clip_max = float(engine_cfg.adv_clip_max)
        clipped_advantages = torch.clamp(slice_advantages, -adv_clip_max, adv_clip_max)
        pred_x_0, _ = self.schedule.convert_from_pred(pred, pos_step_inputs.xts, pos_step_inputs.timesteps)  # get_endpoint
        score_grad_coeff = self.sampler.transition_score_grad_coeff(pred, pos_step_inputs.xts, pos_step_inputs.timesteps, s)

        # RAVEN: grpo.py:586-615
        policy_loss_scaling = float(engine_cfg.policy_loss_scaling)
        policy_losses = []
        policy_grad_norm_values = []
        for step_idx, advantage, timestep_weight, x0, coeff, next_xt_i, mean_i, std_i in zip(
            slice_step_indices,
            clipped_advantages,
            slice_timestep_weights,
            pred_x_0,
            score_grad_coeff,
            next_xt,
            mean,
            std,
        ):
            x0 = x0.float()
            std_i = std_i.clamp_min(1e-6)
            score_grad = 0.5 * coeff.detach() * (-advantage).detach() * (
                (next_xt_i - mean_i).detach() / std_i.detach().pow(2)
            )
            policy_grad = (
                2.0 * score_grad.detach()
                * float(step_idx < num_policy_steps)
                * timestep_weight.detach().to(dtype=score_grad.dtype)
                * policy_loss_scaling
            )
            target = x0.detach() - score_grad.detach()
            policy_losses.append(
                F.mse_loss(x0, target.float(), reduction="none").to(dtype=x0.dtype)
                * float(step_idx < num_policy_steps)
                * timestep_weight.detach().to(dtype=x0.dtype)
            )
            policy_grad_norm_values.append(policy_grad.reshape(-1).norm().to(dtype=torch.float32))

        # RAVEN: grpo.py:616-617
        policy_loss = self._loss_fn(pos_step_inputs, policy_losses, key_prefix="policy_losses")
        total_loss = policy_loss * policy_loss_scaling

        # RAVEN: grpo.py:619-633 — the upstream meter calls carry cnt=batch_size;
        # common.meter.AverageMeter has no count-weighted put_scalar, so the
        # per-observation average is kept instead (metrics only, loss unaffected).
        get_running_average_meter().put_scalar("running/policy_loss", policy_loss.detach())
        get_running_average_meter().put_scalar("running/policy_timestep_weight", slice_timestep_weights.detach().mean())
        get_running_average_meter().put_scalar(
            "running/policy_loss/x0_grad_norm", torch.stack(policy_grad_norm_values).mean()
        )

        # RAVEN: grpo.py:635-663 (KL anchor; ref_mean under no-grad kernel)
        beta = float(getattr(engine_cfg, "beta", 0.0))
        if beta > 0.0:
            with torch.no_grad():
                ref_mean, _ = self.sampler.transition_kernel(
                    ctx["ref_pred"], pos_step_inputs.xts, pos_step_inputs.timesteps, s
                )
            kl_losses = []
            for x0, coeff, mean_i, ref_mean_i, std_i in zip(
                pred_x_0,
                score_grad_coeff,
                mean,
                ref_mean,
                std,
            ):
                x0 = x0.float()
                kl_score_grad = 0.5 * coeff.detach() * (
                    (mean_i - ref_mean_i).detach() / std_i.detach().pow(2)
                )
                kl_target = x0.detach() - kl_score_grad.detach()
                kl_losses.append(
                    F.mse_loss(x0, kl_target.float(), reduction="none").to(dtype=x0.dtype)
                )
            kl_loss = self._loss_fn(pos_step_inputs, kl_losses, key_prefix="kl_losses")
            total_loss = total_loss + beta * kl_loss
            get_running_average_meter().put_scalar("running/kl_loss", kl_loss.detach())

        ctx["policy_loss"] = total_loss
        return ctx

    def merge_tensors(self, values):
        # RAVEN: base_meta_model.py:168-178 (concat is the identity for this meta
        # model, so list-of-lists flattens with sum(values, []))
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
        if isinstance(first, list):
            return sum(values, [])
        raise TypeError(f"Unsupported values type: {type(first)}")

    @staticmethod
    def mask_tensors(values, mask):
        # RAVEN: base_meta_model.py:179-195
        if isinstance(mask, torch.Tensor):
            assert mask.ndim == 1, f"Mask should be 1D, got shape {tuple(mask.shape)}"
            mask = mask.to(dtype=torch.bool)
        else:
            mask = torch.tensor(mask, dtype=torch.bool)

        if isinstance(values, torch.Tensor):
            return values[mask.to(device=values.device)]
        if isinstance(values, list):
            mask_list = mask.tolist()
            return [value for value, keep in zip(values, mask_list) if keep]
        raise TypeError(f"Unsupported values type: {type(values)}")

    def merge_inputs(self, *inputs: ForwardInput, detach: bool = True) -> ForwardInput:
        """Merge per-sample fields; packed index tensors are repacked afterwards."""
        # RAVEN: base_meta_model.py:196-221 (field merge) + wan2_1_t2v.py:98-103
        # (max_seq_len) + causal_wan2_1_t2v.py:285-302 (packed repack).
        assert len(inputs) > 0, "At least one input is required"
        merged = {}
        for f in fields(inputs[0]):
            values = [getattr(input_i, f.name) for input_i in inputs]
            if all(isinstance(v, torch.Tensor) and v.ndim > 0 for v in values):
                merged[f.name] = torch.cat(values, dim=0)
            elif all(isinstance(v, list) for v in values):
                merged[f.name] = sum(values, [])
        merged_inputs = ForwardInput(
            **merged,
            batch_size=sum(input_i.batch_size for input_i in inputs),
            max_seq_len=max(input_i.max_seq_len for input_i in inputs),
        )
        if inputs[0].split_lens is not None:
            unpacked_samples = []
            for input_i in inputs:
                unpacked_samples.extend(self._unpack_packed_samples(input_i))
            merged_inputs = self._repack_from_samples(merged_inputs, unpacked_samples)
        if detach:
            merged_inputs = self._clone_inputs(merged_inputs)
        return merged_inputs

    def mask_inputs(self, inputs: ForwardInput, mask, detach: bool = True) -> ForwardInput:
        """Keep the masked samples; packed index tensors are repacked afterwards."""
        # RAVEN: base_meta_model.py:223-260 (field mask) + wan2_1_t2v.py:105-110
        # (max_seq_len recompute) + causal_wan2_1_t2v.py:304-327 (packed repack).
        batch_size = int(inputs.batch_size.item())
        if isinstance(mask, torch.Tensor):
            assert mask.ndim == 1, f"Mask should be 1D, got shape {tuple(mask.shape)}"
            mask = mask.to(device=inputs.batch_size.device, dtype=torch.bool)
        else:
            mask = torch.tensor(mask, device=inputs.batch_size.device, dtype=torch.bool)
        assert mask.numel() == batch_size, f"Mask length {mask.numel()} does not match batch size {batch_size}"

        kept = int(mask.sum().item())
        mask_list = mask.tolist()
        masked = {}
        for f in fields(inputs):
            value = getattr(inputs, f.name)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == batch_size:
                masked[f.name] = value[mask.to(device=value.device)]
            elif isinstance(value, list) and len(value) == batch_size:
                masked[f.name] = [v for v, keep in zip(value, mask_list) if keep]
        masked_inputs = ForwardInput(**masked, batch_size=inputs.batch_size.new_tensor(kept))
        masked_inputs.max_seq_len = int(masked_inputs.seqlens.max().item()) if kept > 0 else 0
        if inputs.split_lens is not None:
            unpacked_samples = self._unpack_packed_samples(inputs)
            unpacked_samples = [sample for sample, keep in zip(unpacked_samples, mask_list) if keep]
            masked_inputs = self._repack_from_samples(masked_inputs, unpacked_samples)
        if detach:
            masked_inputs = self._clone_inputs(masked_inputs)
        return masked_inputs

    def _unpack_packed_samples(self, inputs: ForwardInput) -> list[dict[str, Any]]:
        """Split packed per-batch index tensors back into per-sample dicts."""
        # RAVEN: causal_wan2_1_t2v.py:124-226. The i2v img-cache assert is dropped
        # (ForwardInput pruned past_key_values_cross_attn_img for this project).
        assert inputs.split_lens is not None and inputs.attn_modes is not None and inputs.sample_lens is not None, \
            "Training packed fields must exist before unpacking"
        assert inputs.past_key_values_self_attn is None and inputs.past_key_values_cross_attn is None, \
            "Unpacking cached inference inputs is not supported"

        batch_size = int(inputs.batch_size.item())
        device = inputs.seqlens.device

        sample_lens_cumsum = torch.cumsum(
            torch.tensor([0] + inputs.sample_lens, dtype=torch.int32, device=device),
            dim=0
        )
        sample_token_indexes = [
            torch.arange(sample_lens_cumsum[i], sample_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            for i in range(batch_size)
        ]

        sample_latent_masks = [torch.isin(inputs.packed_latent_indexes, sample_token_indexes[i]) for i in range(batch_size)]
        latent_lens = [int(mask.sum().item()) for mask in sample_latent_masks]
        latent_lens_cumsum = torch.cumsum(
            torch.tensor([0] + latent_lens, dtype=torch.int32, device=device),
            dim=0
        )

        packed_latent_block_starts = torch.cumsum(inputs.packed_latent_seqlens, dim=0) - inputs.packed_latent_seqlens
        packed_latent_block_sample_indexes = inputs.packed_latent_indexes[packed_latent_block_starts.long()]
        sample_latent_block_masks = [torch.isin(packed_latent_block_sample_indexes, sample_token_indexes[i]) for i in range(batch_size)]
        latent_block_counts = [int(mask.sum().item()) for mask in sample_latent_block_masks]

        packed_noisy_block_starts = torch.cumsum(inputs.packed_noisy_latent_seqlens, dim=0) - inputs.packed_noisy_latent_seqlens
        packed_noisy_block_latent_indexes = inputs.packed_noisy_latent_relative_indexes[packed_noisy_block_starts.long()]
        sample_latent_indexes_per_sample = [
            inputs.packed_latent_indexes[sample_latent_masks[i]] - sample_lens_cumsum[i]
            for i in range(batch_size)
        ]
        noisy_block_counts = []
        for i in range(batch_size):
            sample_noisy_block_mask = torch.isin(
                packed_noisy_block_latent_indexes,
                torch.arange(latent_lens_cumsum[i], latent_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            )
            noisy_block_counts.append(int(sample_noisy_block_mask.sum().item()))

        latent_block_cumsum = torch.cumsum(torch.tensor([0] + latent_block_counts, dtype=torch.int32, device=device), dim=0)
        noisy_block_cumsum = torch.cumsum(torch.tensor([0] + noisy_block_counts, dtype=torch.int32, device=device), dim=0)

        sample_q_range_masks = [
            torch.isin(inputs.q_ranges[:, 0], sample_token_indexes[i]) if inputs.q_ranges is not None else
            torch.zeros((0,), dtype=torch.bool, device=device)
            for i in range(batch_size)
        ]

        unpacked_samples = []
        for i in range(batch_size):
            sample_position_ids = inputs.packed_position_ids[sample_lens_cumsum[i]:sample_lens_cumsum[i + 1]] if inputs.packed_position_ids is not None else \
                torch.zeros((0,), dtype=torch.int32, device=device)

            sample_latent_indexes = sample_latent_indexes_per_sample[i]
            sample_latent_seqlens = inputs.packed_latent_seqlens[latent_block_cumsum[i]:latent_block_cumsum[i + 1]]

            sample_noisy_relative_mask = torch.isin(
                inputs.packed_noisy_latent_relative_indexes,
                torch.arange(latent_lens_cumsum[i], latent_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            )
            sample_noisy_relative_indexes = inputs.packed_noisy_latent_relative_indexes[sample_noisy_relative_mask] - latent_lens_cumsum[i]
            sample_noisy_latent_seqlens = inputs.packed_noisy_latent_seqlens[noisy_block_cumsum[i]:noisy_block_cumsum[i + 1]]

            sample_q_ranges = inputs.q_ranges[sample_q_range_masks[i]] - sample_lens_cumsum[i]
            sample_k_ranges = inputs.k_ranges[sample_q_range_masks[i]] - sample_lens_cumsum[i]
            sample_attn_type_map = inputs.attn_type_map[sample_q_range_masks[i]]
            sample_attn_workload = inputs.attn_workloads[i]

            assert sample_latent_indexes.numel() == int(sample_latent_seqlens.sum().item()), \
                f"Sample {i} latent indexes/seqlens mismatch: {sample_latent_indexes.numel()} vs {int(sample_latent_seqlens.sum().item())}"
            assert sample_noisy_relative_indexes.numel() == int(sample_noisy_latent_seqlens.sum().item()), \
                f"Sample {i} noisy indexes/seqlens mismatch: {sample_noisy_relative_indexes.numel()} vs {int(sample_noisy_latent_seqlens.sum().item())}"

            unpacked_samples.append(dict(
                position_ids=sample_position_ids,
                latent_indexes=sample_latent_indexes,
                latent_seqlens=sample_latent_seqlens,
                noisy_latent_relative_indexes=sample_noisy_relative_indexes,
                noisy_latent_seqlens=sample_noisy_latent_seqlens,
                q_ranges=sample_q_ranges,
                k_ranges=sample_k_ranges,
                attn_type_map=sample_attn_type_map,
                attn_workload=sample_attn_workload,
                split_lens=inputs.split_lens[i],
                attn_modes=inputs.attn_modes[i],
                sample_lens=inputs.sample_lens[i],
                frame_shifts=inputs.frame_shifts[i],
                chunk_sizes=inputs.chunk_sizes[i],
                independent_first_chunks=inputs.independent_first_chunks[i],
                sinks=inputs.sinks[i],
                window_sizes=inputs.window_sizes[i],
            ))

        return unpacked_samples

    def _repack_from_samples(self, inputs: ForwardInput, unpacked_samples: list[dict[str, Any]]) -> ForwardInput:
        """Rebuild packed per-batch index tensors from per-sample dicts."""
        # RAVEN: causal_wan2_1_t2v.py:227-284. RAVEN calls inputs.update(dict(...))
        # on its dict-backed dataclass; ours is a plain dataclass, so the packed
        # fields are assigned directly (same fields, same values).
        device = inputs.seqlens.device

        sample_lens = [sample["sample_lens"] for sample in unpacked_samples]
        sample_lens_cumsum = torch.cumsum(torch.tensor([0] + sample_lens, dtype=torch.int32, device=device), dim=0)[:-1]

        packed_position_ids = torch.cat([sample["position_ids"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)
        packed_latent_indexes = torch.cat([
            sample["latent_indexes"] + sample_lens_cumsum[i] for i, sample in enumerate(unpacked_samples)
        ], dim=0) if len(unpacked_samples) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        packed_latent_seqlens = torch.cat([sample["latent_seqlens"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)
        packed_noisy_latent_seqlens = torch.cat([sample["noisy_latent_seqlens"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)

        q_ranges = []
        k_ranges = []
        attn_type_map = []
        attn_workloads = []
        for i, sample in enumerate(unpacked_samples):
            q_ranges.append(sample["q_ranges"] + sample_lens_cumsum[i])
            k_ranges.append(sample["k_ranges"] + sample_lens_cumsum[i])
            attn_type_map.append(sample["attn_type_map"])
            attn_workloads.append(sample["attn_workload"])
        q_ranges = torch.cat(q_ranges, dim=0) if len(q_ranges) > 0 else torch.zeros((0, 2), dtype=torch.int32, device=device)
        k_ranges = torch.cat(k_ranges, dim=0) if len(k_ranges) > 0 else torch.zeros((0, 2), dtype=torch.int32, device=device)
        attn_type_map = torch.cat(attn_type_map, dim=0) if len(attn_type_map) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)

        latent_offsets = [0]
        for sample in unpacked_samples:
            latent_offsets.append(latent_offsets[-1] + sample["latent_indexes"].numel())
        packed_noisy_latent_relative_indexes = torch.cat([
            sample["noisy_latent_relative_indexes"] + latent_offsets[i] for i, sample in enumerate(unpacked_samples)
        ], dim=0) if len(unpacked_samples) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)

        assert packed_latent_indexes.numel() == int(packed_latent_seqlens.sum().item()), \
            f"Mismatched packed latent indexes and seqlens: {packed_latent_indexes.numel()} vs {int(packed_latent_seqlens.sum().item())}"
        assert packed_noisy_latent_relative_indexes.numel() == int(packed_noisy_latent_seqlens.sum().item()), \
            f"Mismatched packed noisy relative indexes and seqlens: {packed_noisy_latent_relative_indexes.numel()} vs {int(packed_noisy_latent_seqlens.sum().item())}"

        inputs.packed_position_ids = packed_position_ids
        inputs.packed_latent_indexes = packed_latent_indexes
        inputs.packed_latent_seqlens = packed_latent_seqlens
        inputs.packed_noisy_latent_relative_indexes = packed_noisy_latent_relative_indexes
        inputs.packed_noisy_latent_seqlens = packed_noisy_latent_seqlens
        inputs.q_ranges = q_ranges
        inputs.k_ranges = k_ranges
        inputs.attn_type_map = attn_type_map
        inputs.attn_workloads = attn_workloads
        return inputs

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_latents(vae: Any, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        # RAVEN: wan2_1_t2v.py:92-95 (decode_latents). Two mechanical adaptations:
        # the vae arrives via ctx["models"] instead of self.vae, and the dtype attr
        # is param_dtype (renamed during the wan2_1 port).
        return [
            vae.decode(u.unsqueeze(0).to(vae.param_dtype)).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]

    def _reward_fn(self, *, models: dict[str, Any], inputs: ForwardInput, tensors):
        """Decode latents when needed, then dispatch to the (sub) reward model(s)."""
        # RAVEN: causal_wan2_1_t2v.py:1077-1102 (decode branch) +
        # base_meta_model.py:445-457 (BaseRewardModel fanout + merge).
        reward_model = models["reward_model"]
        samples = list(tensors) if isinstance(tensors, torch.Tensor) else tensors
        assert len(samples) > 0, "reward_fn got empty tensors"

        channels = samples[0].shape[0]
        if channels == self.z_dim:
            video_tensors = self._decode_latents(models["vae"], samples)
        elif channels == 3:
            video_tensors = samples
        else:
            raise ValueError(f"Unsupported tensor shape for reward_fn: {tuple(samples[0].shape)}")

        source_fps = 16.0
        if type(reward_model) is BaseRewardModel:
            reward_outputs = []
            for model_name, model in BaseRewardModel.collect_sub_reward_models(models):
                reward_outputs.append((model_name, model(video_tensors, inputs.prompts, source_fps=source_fps)))
            return BaseRewardModel.merge_reward_outputs(reward_outputs)
        return reward_model(video_tensors, inputs.prompts, source_fps=source_fps)
