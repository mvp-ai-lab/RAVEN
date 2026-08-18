"""CausalWanT2VDMD: all DMD algorithm primitives for causal Wan2.1 T2V.

One project == one engine, so this class has NO genericity: it inherits only the
causal-Wan machinery it shares byte-for-byte with ``CausalWanT2VGRPO``
(``CausalWanT2V``); the public surface is the training ctx primitives plus
validation; everything else is a private recipe.

Primitive flow: rollout → FAKE phase → GEN phase.

ctx contract::

    prepare_inputs   reads batch/models/rng/config      writes inputs, neg_inputs
    sync_inputs      reads inputs/neg_inputs            yields synchronized input ctx copies
    rollout          reads models/inputs/neg_inputs/rng writes rollout_x0s, trajectory_xts   (no-grad)
    prepare_fake     reads rollout_x0s/rng              writes fake_timesteps, fake_noises, fake_noisy_latents, fake_inputs
    fake_forward     reads models/fake_inputs           writes fake_pred
    fake_loss        reads fake_pred/fake_inputs        writes fake_loss
    prepare_gen      reads trajectory_xts/rng           writes gen_timesteps, gen_index, gen_xts, score_timesteps, gen_inputs
    gen_forward      reads models/gen_inputs/neg_inputs writes gen_pred                       (graph attached)
    score            reads models/gen_pred/score_t/rng  writes gen_x0s, fake_score_x0s, real_score_x0s
    gen_loss         reads gen_x0s/*_score_x0s          writes gen_loss

``fake_inputs``/``gen_inputs`` are prepared clones of ``inputs`` (single writer
each; read by the corresponding forward/loss primitives). ``fake_metrics`` /
``gen_metrics`` are intentionally left unset: the running meter singletons
(``common.meter``) carry all detail metrics.

Expected configuration shape::

    meta_model:
      dummy_latents: true
      guidance_min: 5.0
      guidance_max: 5.0
      default_neg_prompt: "..."
      chunk_wise_weighting: null
      fake_input: renoise
      gen_input: trajectory
      fake_grad_enabled: false
      real_grad_enabled: false
      fake_loss_type: v_lerp
      dmd_loss:
        type: dmd
        norm_clip_min: 1.0e-5
        norm_per_chunk: false
        phuber_c: 0.001
        alpha: 1.0

Note: ``config.diffusion`` provides 7 components: fake_training_timesteps,
score_timesteps, sampling_timesteps, schedule, fake_schedule, tea_schedule,
sampler.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import fields
from typing import Any, Iterator

import torch

from common.distributed.ops import get_device
from common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from common.logging import get_logger
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, execution_phase
from common.seed import local_seed, yield_seed

from .causal_wan_t2v import CausalWanT2V, ForwardInput

logger = get_logger()


class CausalWanT2VDMD(CausalWanT2V):
    """See module docstring for the ctx contract."""

    def __init__(self, config):
        super().__init__(config)
        mm = config.meta_model

        assert mm.fake_input == "renoise", "only 'renoise' is implemented"
        assert mm.gen_input == "trajectory", "only 'trajectory' is implemented"
        self.fake_grad_enabled = bool(mm.get("fake_grad_enabled", False))
        self.real_grad_enabled = bool(mm.get("real_grad_enabled", False))
        assert mm.fake_loss_type in {"x_0", "x_T", "v_cos", "v_lerp"}, (
            f"unsupported fake_loss_type: {mm.fake_loss_type}"
        )
        self.fake_loss_type = mm.fake_loss_type
        assert mm.dmd_loss.type in {"dmd", "sim", "sid"}, (
            f"unsupported dmd_loss.type: {mm.dmd_loss.type}"
        )
        self.dmd_loss_type = mm.dmd_loss.type
        norm_clip_min = mm.dmd_loss.get("norm_clip_min", 1.0e-5)
        self.norm_clip_min = None if norm_clip_min is None else float(norm_clip_min)
        self.norm_per_chunk = bool(mm.dmd_loss.get("norm_per_chunk", False))
        self.phuber_c = float(mm.dmd_loss.get("phuber_c", 0.001))
        self.sid_alpha = float(mm.dmd_loss.get("alpha", 1.0))
        assert self.phuber_c > 0, "dmd_loss.phuber_c must be positive"

        self.fake_training_timesteps = self._diffusion["fake_training_timesteps"]
        self.score_timesteps = self._diffusion["score_timesteps"]
        self.fake_schedule = self._diffusion["fake_schedule"]
        self.tea_schedule = self._diffusion["tea_schedule"]

    @contextmanager
    def gen_model_context(self, models: dict[str, Any]):
        """Manage model state across the complete GEN forward/backward phase.

        With score-through-fake gradients enabled, freezing fake model parameters
        preserves gradients to its inputs without accumulating fake parameter
        gradients. The engine must keep this context active through GEN optimize.
        """
        if not self.fake_grad_enabled:
            yield
            return

        fake_model = models["fake_model"]
        trainable_names = {
            name for name, parameter in fake_model.named_parameters() if parameter.requires_grad
        }
        for parameter in fake_model.parameters():
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for name, parameter in fake_model.named_parameters():
                parameter.requires_grad_(name in trainable_names)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Reads: inputs, neg_inputs. Writes: yielded ctx copies with synced inputs."""
        inputs = ctx["inputs"]
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return

        neg_inputs = ctx["neg_inputs"]
        input_type = type(inputs)
        neg_input_type = type(neg_inputs)
        payload = (
            {item.name: getattr(inputs, item.name) for item in fields(inputs)},
            {item.name: getattr(neg_inputs, item.name) for item in fields(neg_inputs)},
        )
        sync = SPDistForward(name="dmd_inputs", comm_shape=True, device=inputs.batch_size.device)
        for synced_inputs, synced_neg_inputs in sync(payload):
            sub_ctx = dict(ctx)
            sub_ctx["inputs"] = input_type(**synced_inputs)
            sub_ctx["neg_inputs"] = neg_input_type(**synced_neg_inputs)
            yield sub_ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, neg_inputs, rng. Writes: rollout_x0s, trajectory_xts. (no-grad)"""
        # RAVEN engine training_loop:156-186 — one slice == the whole bs-per-rank batch.
        models = ctx["models"]
        rng = ctx["rng"]
        backbone = models["backbone"]
        device = get_device()

        # Deep-copy at tensor level FIRST so ctx["inputs"]/ctx["neg_inputs"] stay
        # pristine for the fake/gen phases (RAVEN: deepcopy_with_tensor).
        bwd_pos = self._clone_inputs(ctx["inputs"])
        bwd_neg = self._clone_inputs(ctx["neg_inputs"])

        self.sampling_timesteps.set_timesteps(seqlen=bwd_pos.seqlens, device=device)

        # sample_noises — RAVEN: base_meta_model.py:315-324. Per-sample randn from
        # rng BEFORE infer; infer's internal draws follow (exact rng stream order).
        sampling_noises = [
            torch.empty_like(latent).normal_(generator=rng.torch_cuda_generator)
            for latent in bwd_pos.latents
        ]
        bwd_pos.noises = sampling_noises
        bwd_neg.noises = sampling_noises  # SAME noises on neg (RAVEN engine semantics)

        latent_x0s, trajectory_xt, trajectory_pred = self._infer(
            model=backbone,
            rng=rng,
            pos_inputs=bwd_pos,
            neg_inputs=bwd_neg,
        )

        ctx["rollout_x0s"] = latent_x0s
        ctx["trajectory_xts"] = trajectory_xt
        # trajectory_pred is produced by the transcribed infer but intentionally NOT
        # written to ctx: nothing downstream reads it in this trial's config
        # (gen_input == "trajectory" consumes trajectory_xts only).
        del trajectory_pred
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: rollout_x0s, inputs, rng. Writes: fake_timesteps, fake_noises,
        fake_noisy_latents, fake_inputs."""
        # RAVEN: dmd.py:261-376 (fake_random_noise=True branch only; uncond_fake_train_prob=0
        # branch at dmd.py:346-348 is dead — no rng draw, do not port)
        latent_x0s = ctx["rollout_x0s"]
        rng = ctx["rng"]
        # each phase works on its own tensor-level copy of the inputs (RAVEN semantics)
        fake_inputs = self._clone_inputs(ctx["inputs"])

        # draw 1 — RAVEN: dmd.py:275
        fake_timesteps = self._sample_timesteps(fake_inputs, self.fake_training_timesteps, rng)

        # fake_random_noise branch — RAVEN: dmd.py:332-338
        fake_inputs.latents = latent_x0s  # set_latents (base_meta_model.py:266-272)
        # draw 2 — sample_noises (base_meta_model.py:315-324)
        fake_noises = [
            torch.empty_like(latent).normal_(generator=rng.torch_cuda_generator)
            for latent in fake_inputs.latents
        ]
        fake_inputs.noises = fake_noises
        fake_inputs.timesteps = fake_timesteps
        # add_noises — RAVEN: base_meta_model.py:326-336 (no rng)
        fake_noisy_latents = self.fake_schedule.forward(
            x_0=fake_inputs.latents, x_T=fake_inputs.noises, t=fake_inputs.timesteps
        )
        fake_inputs.xts = fake_noisy_latents

        ctx["fake_timesteps"] = fake_timesteps
        ctx["fake_noises"] = fake_noises
        ctx["fake_noisy_latents"] = fake_noisy_latents
        ctx["fake_inputs"] = fake_inputs
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, fake_inputs. Writes: fake_pred."""
        # fake_model is a NON-causal WanModel: plain batched pred, not the packed
        # causal path. FSDP unshard prefetches (RAVEN: dmd.py:272-273) are
        # perf-only mechanics — skipped.
        ctx["fake_pred"] = self._pred_flat(ctx["models"]["fake_model"], ctx["fake_inputs"])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: fake_pred, fake_inputs. Writes: fake_loss."""
        # RAVEN: dmd.py:363-375
        fake_inputs = ctx["fake_inputs"]
        # convert_pred — RAVEN: base_meta_model.py:392-409
        pred_x_0, pred_x_T = self.fake_schedule.convert_from_pred(
            pred=ctx["fake_pred"], x_t=fake_inputs.xts, t=fake_inputs.timesteps
        )
        pred = self.fake_schedule.convert_to_pred(
            x_0=pred_x_0, x_T=pred_x_T, t=fake_inputs.timesteps, pred_type=self.fake_loss_type
        )
        # convert_target — RAVEN: base_meta_model.py:411-422
        target = self.fake_schedule.convert_to_pred(
            x_0=fake_inputs.latents,
            x_T=fake_inputs.noises,
            t=fake_inputs.timesteps,
            pred_type=self.fake_loss_type,
        )
        ctx["fake_loss"] = self._loss_fn(fake_inputs, pred, target, key_prefix="fake_losses")
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: trajectory_xts, rollout_x0s, inputs, rng. Writes: gen_timesteps,
        gen_index, gen_xts, score_timesteps, gen_inputs."""
        # RAVEN: dmd.py:378-416 (gen_use_trajectory=True, score_timesteps present)
        gen_inputs = self._clone_inputs(ctx["inputs"])
        trajectory_xt = ctx["trajectory_xts"]
        rng = ctx["rng"]
        latent_x0s = ctx["rollout_x0s"]
        bs = int(gen_inputs.batch_size)

        # RAVEN: dmd.py:391-395
        self.sampling_timesteps.set_timesteps(seqlen=gen_inputs.seqlens, device=get_device())
        gen_timesteps = self._sample_timesteps(gen_inputs, self.sampling_timesteps, rng)  # draw 1
        random_index = self.sampling_timesteps.index(gen_timesteps)
        # draw 2 — RAVEN passes fake_inputs here; same seqlens, hence identical draw
        score_timesteps = self._sample_timesteps(gen_inputs, self.score_timesteps, rng)

        # RAVEN: dmd.py:409-416
        gen_inputs.latents = latent_x0s
        gen_inputs.timesteps = gen_timesteps
        gen_xts = [trajectory_xt[random_index[i]][i] for i in range(bs)]
        gen_inputs.xts = gen_xts

        ctx["gen_timesteps"] = gen_timesteps
        ctx["gen_index"] = random_index
        ctx["gen_xts"] = gen_xts
        ctx["score_timesteps"] = score_timesteps
        ctx["gen_inputs"] = gen_inputs
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_inputs, neg_inputs. Writes: gen_pred. (graph attached)"""
        # RAVEN: dmd.py:427 — causal packed path, GRAD ON
        ctx["gen_pred"] = self._pred(ctx["models"]["backbone"], ctx["gen_inputs"], ctx["neg_inputs"])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def score(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_pred, gen_inputs, score_timesteps, rng.
        Writes: gen_x0s, fake_score_x0s, real_score_x0s."""
        # RAVEN: dmd.py:428-461
        models = ctx["models"]
        rng = ctx["rng"]
        gen_inputs = ctx["gen_inputs"]
        gen_pred = ctx["gen_pred"]
        latent_x0s = ctx["rollout_x0s"]
        score_timesteps = ctx["score_timesteps"]

        # get_endpoint — RAVEN: base_meta_model.py:383-390. Grad flows, NO detach.
        gen_x0s, _ = self.schedule.convert_from_pred(gen_pred, gen_inputs.xts, gen_inputs.timesteps)

        # consistency renoise: computed OUTSIDE no_grad — the graph flows through
        # step_to's endpoint into score_xts (RAVEN: dmd.py:429). draw 1.
        score_xts = self.sampler.step_to(
            pred=gen_pred, x_t=gen_inputs.xts, t=gen_inputs.timesteps,
            s=score_timesteps, rng=rng, seqlens=gen_inputs.seqlens,
        )

        # fake score — RAVEN: dmd.py:438-447
        # FSDP unshard prefetches (RAVEN: dmd.py:435-436, 449-450) skipped: perf-only.
        fake_inputs = self._clone_inputs(ctx["inputs"])
        fake_inputs.latents = latent_x0s
        fake_inputs.xts = score_xts
        fake_inputs.timesteps = score_timesteps
        fake_ctx = nullcontext() if self.fake_grad_enabled else torch.no_grad()
        with fake_ctx:
            fake_pred = self._pred_flat(models["fake_model"], fake_inputs)
            fake_score_x0s, _ = self.fake_schedule.convert_from_pred(
                fake_pred, fake_inputs.xts, fake_inputs.timesteps
            )

        # real score via teacher CFG — RAVEN: dmd.py:452-461
        real_pos = self._clone_inputs(ctx["inputs"])
        real_neg = self._clone_inputs(ctx["neg_inputs"])
        for inp in (real_pos, real_neg):
            inp.latents = latent_x0s
            inp.xts = score_xts
            inp.timesteps = score_timesteps
        real_ctx = nullcontext() if self.real_grad_enabled else torch.no_grad()
        with real_ctx:
            real_pred = self._pred_cfg_flat(models["tea_model"], real_pos, real_neg)
            real_score_x0s, _ = self.tea_schedule.convert_from_pred(
                real_pred, real_pos.xts, real_pos.timesteps
            )

        ctx["gen_x0s"] = gen_x0s
        ctx["fake_score_x0s"] = fake_score_x0s
        ctx["real_score_x0s"] = real_score_x0s
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: gen_x0s, fake_score_x0s, real_score_x0s, gen_inputs. Writes: gen_loss."""
        # RAVEN: dmd.py:464-488
        gen_inputs = ctx["gen_inputs"]
        losses = []
        for i, (x0, fake, real) in enumerate(
            zip(ctx["gen_x0s"], ctx["fake_score_x0s"], ctx["real_score_x0s"])
        ):
            x0, fake, real = x0.double(), fake.double(), real.double()
            if self.dmd_loss_type == "sim":
                diff = real - fake
                norm = torch.sqrt((diff**2).sum() + self.phuber_c**2)
                loss = (real - fake) * (fake - x0) / norm
            else:
                if self.norm_per_chunk:
                    # LongLive2.0 model/dmd.py:141-171 ("Changed the normalizer for causal
                    # teacher - per-block normalization"): each chunk is divided by its own
                    # |x0 - real| mean instead of one scalar over the whole rollout, so a
                    # chunk's scale cannot be set by the drift accumulated in later chunks.
                    first_chunk_size = (
                        gen_inputs.independent_first_chunks[i]
                        if gen_inputs.independent_first_chunks[i] is not None
                        else gen_inputs.chunk_sizes[i]
                    )
                    chunk_sizes = [first_chunk_size]
                    t_rest = x0.size(1) - first_chunk_size
                    assert t_rest % gen_inputs.chunk_sizes[i] == 0, (
                        f"t_rest {t_rest} not divisible by chunk size {gen_inputs.chunk_sizes[i]}"
                    )
                    chunk_sizes.extend([gen_inputs.chunk_sizes[i]] * (t_rest // gen_inputs.chunk_sizes[i]))
                    norm = torch.empty_like(x0)
                    start_idx = 0
                    for chunk_size in chunk_sizes:
                        end_idx = start_idx + chunk_size
                        norm[:, start_idx:end_idx, ...] = torch.abs(
                            x0[:, start_idx:end_idx, ...] - real[:, start_idx:end_idx, ...]
                        ).mean()
                        start_idx = end_idx
                else:
                    norm = torch.abs(x0 - real).mean()
                if self.norm_clip_min is not None:
                    norm.clamp_min_(self.norm_clip_min)
                if self.dmd_loss_type == "dmd":
                    loss = (real - fake) * (fake - x0) / norm.detach()
                else:
                    loss = (real - fake) * (
                        (real - x0) - self.sid_alpha * (real - fake)
                    ) / norm.detach()
            get_running_average_meter().put_scalar("running/dmd_loss/norm", norm.mean().item())
            losses.append(loss)
        ctx["gen_loss"] = self._loss_fn(
            ctx["gen_inputs"], losses, key_prefix="dmd_losses",
            score_timesteps=ctx["score_timesteps"],
        )
        return ctx

    def _sample_timesteps(self, inputs: ForwardInput, timesteps, rng) -> torch.Tensor:
        """Draw per-sample timesteps from a timesteps component, advancing rng."""
        # RAVEN: base_meta_model.py:298-313. Components sample under local_seed
        # (global RNG), then the seed is threaded forward with yield_seed.
        # 63-bit combine_seed reduced to numpy's range; stream distinctness comes
        # from yield_seed advancement.
        with local_seed(rng.seed % 2**31):
            ts = timesteps.sample(
                size=(int(inputs.batch_size),),
                seqlens=inputs.seqlens,
                device=get_device(),
            )
        rng.seed = yield_seed(rng.seed)
        return ts
