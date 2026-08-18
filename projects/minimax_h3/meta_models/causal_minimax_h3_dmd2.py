# SPDX-License-Identifier: Apache-2.0
"""DMD2 adversarial training for chunk-causal MiniMax-H3."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from common.diffusion import build_diffusion
from common.distributed.ops import get_device
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, execution_phase
from common.seed import local_seed, yield_seed
from projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
    ForwardInput,
    _RolloutX0s,
)


@dataclass(frozen=True)
class _DMD2RolloutX0s(_RolloutX0s):
    """Final rollout for GEN plus random-stage x0s for phase-local FAKE."""

    fake_video: list[Tensor]
    fake_audio: list[Tensor]


class CausalMiniMaxH3DMD2(CausalMiniMaxH3DMD):
    """Add a tapped bidirectional discriminator objective to MiniMax-H3 DMD.

    The discriminator sees corrupted x_t by default, or clean x with a
    pretended random timestep when ``gan_disc_clean_input`` is enabled.
    """

    _carry_clean_latents = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        mm = config.meta_model
        self.gan_lambda_disc = float(mm.get("gan_lambda_disc", 1.0e-2))
        self.gan_lambda_gen = float(mm.get("gan_lambda_gen", 5.0e-3))
        self.gan_disc_clean_input = bool(mm.get("gan_disc_clean_input", False))
        # Seaweed-APT uses lambda=100 and sigma=0.1 for video. Under a
        # first-order Taylor expansion, E[||D(x+sigma*eps)-D(x)||^2] =
        # sigma^2||grad_x D||^2; unlike true R1, it avoids the double backward
        # that is unavailable with FSDP/FlashAttention. APT's discriminator
        # loss carries an implicit lambda_D of 1, while ours pre-scales by
        # gan_lambda_disc (1e-2), so matching the paper's ratio needs
        # gan_r1_lambda ~ 100 * gan_lambda_disc ~ 1.0.
        self.gan_r1_lambda = float(mm.get("gan_r1_lambda", 0.0))
        self.gan_r1_sigma = float(mm.get("gan_r1_sigma", 0.1))
        self.fake_use_trajectory = bool(mm.get("fake_use_trajectory", False))

        gan_diffusion = build_diffusion(
            {"gan_training_timesteps": config.diffusion.gan_training_timesteps}
        )
        self.gan_training_timesteps = gan_diffusion["gan_training_timesteps"]
        if float(self.gan_training_timesteps.T) != float(self.fake_schedule.T):
            raise ValueError("gan_training_timesteps.T must match fake_schedule.T")

    @contextmanager
    def gen_model_context(self, models: dict[str, Any]) -> Iterator[None]:
        """Freeze the fake model and its discriminator while preserving input gradients."""
        frozen = (models["fake_model"],)
        requires_grad = [
            [parameter.requires_grad for parameter in model.parameters()]
            for model in frozen
        ]
        for model in frozen:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        try:
            with super().gen_model_context(models):
                yield
        finally:
            for model, states in zip(frozen, requires_grad, strict=True):
                for parameter, state in zip(model.parameters(), states, strict=True):
                    parameter.requires_grad_(state)

    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        if not self.fake_use_trajectory:
            return super().rollout(ctx)
        with execution_phase(ExecutionPhase.ROLLOUT), torch.no_grad():
            trajectory_x0_chunks: list[
                tuple[list[Tensor], list[Tensor]]
            ] = []
            rollout_x0s, trajectory_xts = self._rollout_latents(
                ctx["models"]["backbone"],
                ctx["inputs"],
                ctx["rng"],
                keep_trajectory=True,
                trajectory_x0_chunks=trajectory_x0_chunks,
            )
            inputs = ctx["inputs"]
            stage_count = int(self.sampling_timesteps.timesteps.numel())
            chunk_count = len(inputs.layouts[0].chunks)
            assert len(trajectory_x0_chunks) == chunk_count * stage_count
            trajectory_timesteps = self._sample_timesteps(
                inputs, self.sampling_timesteps, ctx["rng"]
            )
            indices = self.sampling_timesteps.index(trajectory_timesteps)
            assert bool((indices >= 0).all()), (
                "trajectory timestep is not on the video sampling grid; "
                "index() returned -1"
            )
            stage_indices = [int(value) for value in indices.tolist()]
            fake_video = [
                torch.cat(
                    [
                        trajectory_x0_chunks[
                            chunk_index * stage_count + stage_indices[index]
                        ][0][index]
                        for chunk_index in range(chunk_count)
                    ],
                    dim=1,
                )
                for index in range(inputs.batch_size)
            ]
            fake_audio = [
                torch.cat(
                    [
                        trajectory_x0_chunks[
                            chunk_index * stage_count + stage_indices[index]
                        ][1][index]
                        for chunk_index in range(chunk_count)
                    ],
                    dim=2,
                )
                for index in range(inputs.batch_size)
            ]
            del trajectory_x0_chunks
            # Keep final-stage x0s on the base payload for GEN; prepare_fake swaps
            # only its phase-local ctx to these random-stage predictions.
            ctx["rollout_x0s"] = _DMD2RolloutX0s(
                video=rollout_x0s.video,
                audio=rollout_x0s.audio,
                video_eps=rollout_x0s.video_eps,
                audio_eps=rollout_x0s.audio_eps,
                fake_video=fake_video,
                fake_audio=fake_audio,
            )
            ctx["trajectory_xts"] = trajectory_xts
            return ctx

    def _prepare_gan_inputs(
        self,
        inputs: ForwardInput,
        latents: tuple[list[Tensor], list[Tensor]],
        rng: Any,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], Tensor, Tensor]:
        with local_seed(rng.seed % 2**31):
            drawn = self.gan_training_timesteps.sample_pair(
                (inputs.batch_size,), inputs.seqlens, get_device()
            )
        rng.seed = yield_seed(rng.seed)
        video_timesteps, audio_timesteps = (
            value.to(torch.float32) for value in drawn
        )
        if self.gan_disc_clean_input:
            return latents, video_timesteps, audio_timesteps

        video_noises, audio_noises = self._sample_noises(inputs, rng)
        noisy_latents = (
            self.fake_schedule.forward(
                latents[0], video_noises, video_timesteps
            ),
            self.fake_schedule.forward(
                latents[1], audio_noises, audio_timesteps
            ),
        )
        return noisy_latents, video_timesteps, audio_timesteps

    def _pred_gan_logits(
        self,
        fake_model: Any,
        inputs: ForwardInput,
        *,
        noisy_latents: tuple[list[Tensor], list[Tensor]],
        video_timesteps: Tensor,
        audio_timesteps: Tensor,
    ) -> Tensor:
        """Run the native bidirectional pack and classify its internal taps."""
        device = inputs.token_tags.device
        kwargs, row_timesteps, img_pos, audio_pos = self._bidirectional_kwargs(
            fake_model,
            inputs,
            video_xts=noisy_latents[0],
            audio_xts=noisy_latents[1],
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        assert bool((row_timesteps[img_pos] != 0).all()) and bool(
            (row_timesteps[audio_pos] != 0).all()
        ), "a bidirectional GAN row carries the clean timestep 0"

        document_lens = torch.stack(
            [
                native["cu_seqlens"][1:] - native["cu_seqlens"][:-1]
                for native in inputs.native
            ]
        ).reshape(-1).to(dtype=torch.int32, device=device)
        live_documents = torch.arange(
            document_lens.numel(), device=device
        ).remainder(2).eq(0)
        nonempty_documents = document_lens > 0
        k_lens = document_lens[nonempty_documents]
        live_documents = live_documents[nonempty_documents]
        return fake_model(
            **kwargs,
            classify_mode=True,
            gan_video_timesteps=video_timesteps,
            gan_k_lens=k_lens,
            gan_live_documents=live_documents,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = ctx["inputs"]
        if inputs.clean_latents is None:
            raise ValueError(
                "MiniMax-H3 DMD2 requires corpus video/audio latents for "
                "discriminator real samples"
            )
        if self.fake_use_trajectory:
            trajectory_rollout: _DMD2RolloutX0s = ctx["rollout_x0s"]
            # The engine shallow-copies the base payload per phase. Replacing this
            # key affects both FAKE objectives below but leaves GEN's final x0s.
            ctx["rollout_x0s"] = _RolloutX0s(
                video=trajectory_rollout.fake_video,
                audio=trajectory_rollout.fake_audio,
                video_eps=trajectory_rollout.video_eps,
                audio_eps=trajectory_rollout.audio_eps,
            )
        ctx = super().prepare_fake(ctx)
        rollout: _RolloutX0s = ctx["rollout_x0s"]

        (
            ctx["gan_fake_noisy_latents"],
            ctx["gan_fake_timesteps"],
            ctx["gan_fake_audio_timesteps"],
        ) = self._prepare_gan_inputs(
            inputs, (rollout.video, rollout.audio), ctx["rng"]
        )
        (
            ctx["gan_real_noisy_latents"],
            ctx["gan_real_timesteps"],
            ctx["gan_real_audio_timesteps"],
        ) = self._prepare_gan_inputs(inputs, inputs.clean_latents, ctx["rng"])
        if self.gan_r1_lambda > 0:
            video_noises, audio_noises = self._sample_noises(inputs, ctx["rng"])
            real_video, real_audio = ctx["gan_real_noisy_latents"]
            ctx["gan_real_perturbed_latents"] = (
                [
                    latent + self.gan_r1_sigma * noise
                    for latent, noise in zip(real_video, video_noises, strict=True)
                ],
                [
                    latent + self.gan_r1_sigma * noise
                    for latent, noise in zip(real_audio, audio_noises, strict=True)
                ],
            )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().fake_loss(ctx)
        fake_model = ctx["models"]["fake_model"]
        inputs = ctx["fake_inputs"]
        fake_logits = self._pred_gan_logits(
            fake_model,
            inputs,
            noisy_latents=ctx["gan_fake_noisy_latents"],
            video_timesteps=ctx["gan_fake_timesteps"],
            audio_timesteps=ctx["gan_fake_audio_timesteps"],
        )
        real_logits = self._pred_gan_logits(
            fake_model,
            inputs,
            noisy_latents=ctx["gan_real_noisy_latents"],
            video_timesteps=ctx["gan_real_timesteps"],
            audio_timesteps=ctx["gan_real_audio_timesteps"],
        )
        fake_term = F.softplus(fake_logits).mean(dim=1).mean()
        real_term = F.softplus(-real_logits).mean(dim=1).mean()
        gan_loss = fake_term + real_term
        real_graph_term = self.gan_lambda_disc * real_term
        if self.gan_r1_lambda > 0:
            real_perturbed_logits = self._pred_gan_logits(
                fake_model,
                inputs,
                noisy_latents=ctx["gan_real_perturbed_latents"],
                video_timesteps=ctx["gan_real_timesteps"],
                audio_timesteps=ctx["gan_real_audio_timesteps"],
            )
            ar1_term = (
                (real_logits - real_perturbed_logits).pow(2).mean(dim=1).mean()
            )
            real_graph_term = real_graph_term + self.gan_r1_lambda * ar1_term

        meter = get_running_average_meter()
        for tap_index, value in enumerate(
            fake_logits.mean(dim=0).detach().tolist()
        ):
            meter.put_scalar(
                f"fake_losses/gan_logits_fake_mean/tap{tap_index}", value
            )
        for tap_index, value in enumerate(
            real_logits.mean(dim=0).detach().tolist()
        ):
            meter.put_scalar(
                f"fake_losses/gan_logits_real_mean/tap{tap_index}", value
            )
        meter.put_scalar("fake_losses/gan_loss", float(gan_loss.detach()))
        if self.gan_r1_lambda > 0:
            meter.put_scalar("fake_losses/gan_ar1", float(ar1_term.detach()))
        # THREE terms rather than their sum: score, fake D, and real D. The
        # engine backwards each on its own GraphTask, which keeps the FAKE
        # step's memory bounded -- summing them makes every parameter's
        # AccumulateGrad depend on all subgraphs, and the autograd engine then
        # parks each block's gradient in an InputBuffer until the last arrives.
        # See engines/dmd.py::_fake_backward for the measurements. aR1 cannot
        # become a fourth GraphTask: it shares real_logits with the real term
        # and the engine calls backward without retain_graph, so splitting
        # would raise backward-through-graph a second time. The cost is a
        # second D forward inside the real element (AccumulateGrad dep count
        # 2, the same InputBuffer parking that motivated the split) and one
        # more concurrently-live forward graph.
        #
        # The sum is unchanged, so the parameter update is too: FSDP2 adds into
        # sharded_param.grad rather than overwriting it.
        ctx["fake_loss"] = (
            ctx["fake_loss"],
            self.gan_lambda_disc * fake_term,
            real_graph_term,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().gen_loss(ctx)
        noisy_latents, video_timesteps, audio_timesteps = self._prepare_gan_inputs(
            ctx["gen_inputs"], ctx["gen_x0s"], ctx["rng"]
        )
        logits = self._pred_gan_logits(
            ctx["models"]["fake_model"],
            ctx["gen_inputs"],
            noisy_latents=noisy_latents,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        gan_loss = F.softplus(-logits).mean(dim=1).mean()

        meter = get_running_average_meter()
        for tap_index, value in enumerate(logits.mean(dim=0).detach().tolist()):
            meter.put_scalar(
                f"dmd_losses/gan_logits_mean/tap{tap_index}", value
            )
        meter.put_scalar("dmd_losses/gan_loss", float(gan_loss.detach()))
        ctx["gen_loss"] = ctx["gen_loss"] + self.gan_lambda_gen * gan_loss
        return ctx


__all__ = ["CausalMiniMaxH3DMD2"]
