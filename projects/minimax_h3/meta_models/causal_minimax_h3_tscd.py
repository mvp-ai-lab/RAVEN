# SPDX-License-Identifier: Apache-2.0
"""CausalMiniMaxH3TSCD: trajectory-segmented consistency distillation for chunk-causal MiniMax-H3.

TSCD extends the teacher-forced ``CausalMiniMaxH3TF`` route: every noisy chunk is
still supervised against the corpus's real x0 history, while a frozen causal
teacher supplies one adjacent ODE step and the causal EMA backbone supplies the
self-consistency target. ``engines/tscd.py`` calls ``prepare_inputs`` and
``sync_inputs`` (both inherited), then ``sample_segment`` (draws
``start_timesteps``, ``step_timesteps``, ``boundary_timesteps`` and
``target_timesteps``), ``add_noise``, ``student_forward``, ``solver_step`` and
``target_forward`` (both no-grad), and finally ``compute_loss``.
``config.meta_model`` carries ``audio_loss_weight``, ``num_segments`` and
``loss_type``; ``config.diffusion`` carries the CD training grid
(``sampling_timesteps``, ``audio_sampling_timesteps``, ``schedule``, ``sampler``,
``tea_schedule``, ``tea_sampler``) while ``validation`` may carry its own, usually
far coarser, rollout grid.

Video and audio share each sampled grid index but use their own shifted grids:
H3 aligns denoising steps, not sigmas. The teacher always integrates from the
sampled start to the adjacent grid point, and the segment boundary is used only
to draw the common comparison time. All three model forwards use the causal
packed ABI with corpus x0 context, and independently sampled context eps remains
required because the H3 wrapper perturbs rows whose repo timestep is exactly 0.
``sampling_timesteps`` MUST set ``sampling_skip_max`` to at least 1: the bound
constrains the START index, but what has to stay on the grid is the ODE target
``t_{i+1}`` -- ``get_timesteps_by_index`` returns the clean bound 0 past the end,
and ``target_forward`` would hand that to the EMA, where the ``repo_t == 0`` test
treats the chunk as context and re-noises it against the zero eps the packer gives
noisy rows. With no c_skip/c_out to make ``t' = 0`` meaningful, the last index is
excluded by configuration, so the student never learns that final step -- a real
cost, and a reason to prefer a longer grid.

``models`` needs ``backbone``, ``backbone_ema`` (the engine's EMA target) and
``tea_model``, plus ``text_encoder`` and both VAEs once validation runs. All three
DiT nodes wrap the CAUSAL model -- unlike DMD, where ``tea_model`` wraps the
bidirectional one. The teacher is a chunk-causal checkpoint called through the
same packed causal ABI as the student, so the ODE step it supplies is conditioned
on the history the student actually sees; a bidirectional teacher would put
``x_{t'}`` on a trajectory the student is structurally unable to reach. No CFG
anywhere -- H3 is guidance-distilled. Self-consistency is learned only along
teacher-forced trajectories, since the teacher's ODE step is also taken on corpus
x0 history, which does not exist during free rollout; that is a property of this
two-stage route, not a defect.
"""

from __future__ import annotations

from typing import Any

import torch

from common.diffusion import build_diffusion
from common.diffusion.schedule import PredictionType
from common.distributed.ops import get_device
from common.phase import ExecutionPhase, execution_phase
from common.seed import local_seed, yield_seed
from projects.minimax_h3.meta_models.causal_minimax_h3_tf import CausalMiniMaxH3TF


class CausalMiniMaxH3TSCD(CausalMiniMaxH3TF):
    """Teacher-forced TSCD over corpus x0 context.

    TSCD inherits TF rather than DMD because its clean context still comes from
    the corpus's real x0; it adds self-consistency on top of per-chunk denoising,
    so ``prepare_inputs``, ``sync_inputs`` and ``_noise_chunks`` stay unchanged.
    """

    def __init__(self, config: Any) -> None:
        # Deliberately not super().__init__: the TF and DMD constructors require
        # algorithm-specific diffusion nodes that TSCD does not use.
        self.config = config
        self.audio_loss_weight = float(config.meta_model.audio_loss_weight)
        self.chunk_wise_weighting = None
        self._validation_packer_cache: Any = None

        self.num_segments = int(config.meta_model.num_segments)
        # Not a PredictionType: this picks WHERE the two predictions meet, not
        # what they are reparameterized into. ``x_t`` projects both sides to s
        # and compares noisy latents; ``x_0`` compares the estimates directly.
        # They are NOT weighted versions of each other -- expanding the
        # projection gives c(t)(p_stu - p_tea) - c(t')(p_tgt - p_tea) with
        # c(u) = A_s - B_s*A_u/B_u. As s -> 0 both coefficients go to 1 and it
        # collapses onto the x_0 branch (plain CD); at s = t' the second one
        # vanishes and the target switches from the EMA to the teacher. Those
        # are the two limits engines/tscd.py:46-51 describes.
        self.loss_type = str(config.meta_model.loss_type)
        assert self.loss_type in {"x_t", "x_0"}, (
            f"meta_model.loss_type must be x_t or x_0, got {self.loss_type}"
        )

        diffusion = build_diffusion(config.diffusion)
        self.sampling_timesteps = diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = diffusion["audio_sampling_timesteps"]
        self.schedule = diffusion["schedule"]
        self.sampler = diffusion["sampler"]
        self.tea_schedule = diffusion["tea_schedule"]
        self.tea_sampler = diffusion["tea_sampler"]

        # A consistency student is distilled ON a fine grid and SERVED on a
        # coarse one, so the rollout cannot reuse the pair above: those define
        # the CD training grid. ``validation`` carries its own timesteps and its
        # own sampler (a consistency step jumps to x0 and re-noises, where the
        # teacher's DDIM integrates), and validate() swaps them in.
        #
        # ``schedule`` is deliberately taken from ``diffusion`` rather than
        # duplicated under ``validation``: it is the noising math both sides
        # must agree on, and a second copy in the trial could drift from the one
        # the loss is computed against without anything catching it.
        validation = config.validation
        self.validation_diffusion = build_diffusion(
            {
                "sampling_timesteps": validation.sampling_timesteps,
                "audio_sampling_timesteps": validation.audio_sampling_timesteps,
                "schedule": config.diffusion.schedule,
                "sampler": validation.sampler,
            }
        )

        # tea_schedule is never read here -- build_diffusion injects it into
        # tea_sampler. Asserting it is what keeps a wrong pred_type from
        # silently reinterpreting the teacher's x0 as a velocity inside the one
        # ODE step, which no downstream shape or value would reveal.
        for name in ("schedule", "tea_schedule"):
            pred_type = getattr(self, name).pred_type
            assert pred_type == PredictionType.x_0, (
                f"diffusion.{name}.pred_type must be x_0 because every "
                f"MiniMax-H3 model node returns x0, got {pred_type}"
            )

        # The pairing IS the grid index, so both grids must be indexable by one
        # integer; min/max/segment_length below are derived from the video grid
        # alone and silently apply to audio.
        # Checking dynamic_shift rather than timesteps.dim(): a grid that was
        # never built leaves timesteps as None, and .dim() would raise before
        # this message ever printed. Only the dynamic branch produces 2-D grids.
        assert (
            not self.sampling_timesteps.dynamic_shift
            and not self.audio_sampling_timesteps.dynamic_shift
            and self.sampling_timesteps.num_sampling_steps
            == self.audio_sampling_timesteps.num_sampling_steps
        ), "video and audio sampling grids must be static and share one step count"

        # segment_length = 0 would make the boundary index an integer division
        # by zero, which on CUDA yields garbage rather than raising.
        assert 1 <= self.num_segments <= self.sampling_timesteps.num_sampling_steps, (
            f"meta_model.num_segments must be in [1, "
            f"{self.sampling_timesteps.num_sampling_steps}], got {self.num_segments}"
        )

        # See the module docstring: the ODE target t_{i+1} is fed to the EMA as
        # a real timestep, and index N is the grid's clean bound 0, which the
        # H3 wrapper's exact-equality context test would then re-noise against
        # a zero eps. Clamping max_index here instead would make the trial yaml
        # lie about which part of the grid it trains. Only the video grid is
        # checked: the index draw is video-derived, and the equal-step-count
        # assert above is what carries this over to audio.
        assert (self.sampling_timesteps.sampling_skip_max or 0) >= 1, (
            "diffusion.sampling_timesteps.sampling_skip_max must be >= 1 so the "
            "ODE step target t_{i+1} always lands on the grid"
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_segment(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, rng. Writes: start_timesteps, step_timesteps,
        boundary_timesteps, target_timesteps."""
        inputs = ctx["inputs"]
        rng = ctx["rng"]
        device = get_device()
        chunk_counts = [len(layout.chunks) for layout in inputs.layouts]

        min_index = (
            self.sampling_timesteps.sampling_skip_min
            if self.sampling_timesteps.sampling_skip_min is not None
            else 0
        )
        max_index = self.sampling_timesteps.num_sampling_steps - (
            self.sampling_timesteps.sampling_skip_max
            if self.sampling_timesteps.sampling_skip_max is not None
            else 0
        )
        with local_seed(rng.seed % 2**31):
            indices = torch.randint(
                min_index, max_index, (sum(chunk_counts),), device=device
            )
        rng.seed = yield_seed(rng.seed)

        segment_length = self.sampling_timesteps.num_sampling_steps // self.num_segments
        boundary_indices = (indices // segment_length + 1) * segment_length

        video_start = self.sampling_timesteps.get_timesteps_by_index(indices)
        video_step = self.sampling_timesteps.get_timesteps_by_index(indices + 1)
        video_boundary = self.sampling_timesteps.get_timesteps_by_index(boundary_indices)

        audio_start = self.audio_sampling_timesteps.get_timesteps_by_index(indices)
        audio_step = self.audio_sampling_timesteps.get_timesteps_by_index(indices + 1)
        audio_boundary = self.audio_sampling_timesteps.get_timesteps_by_index(boundary_indices)

        # lerp_random's (0, 1] draw, taken once and shared: two lerp_random
        # calls would consume the stream twice and silently decouple video from
        # audio. Note what is shared here is a linear ratio in sigma space, not
        # the grid-index pairing the start times get -- acceptable because s
        # only ever enters schedule.forward and is never fed to a model; it just
        # sets how noisy the two modalities' comparison points are relative to
        # each other. _generator rather than lerp_random's bare
        # torch_cuda_generator: that one is None without CUDA and would fall
        # back to the global stream, breaking resume.
        weight = 1.0 - torch.rand(
            (sum(chunk_counts),), device=device, generator=self._generator(rng)
        )
        video_target = video_boundary + weight * (video_step - video_boundary)
        audio_target = audio_boundary + weight * (audio_step - audio_boundary)

        ctx["start_timesteps"] = (
            list(video_start.to(torch.float32).split(chunk_counts)),
            list(audio_start.to(torch.float32).split(chunk_counts)),
        )
        ctx["step_timesteps"] = (
            list(video_step.to(torch.float32).split(chunk_counts)),
            list(audio_step.to(torch.float32).split(chunk_counts)),
        )
        ctx["boundary_timesteps"] = (
            list(video_boundary.to(torch.float32).split(chunk_counts)),
            list(audio_boundary.to(torch.float32).split(chunk_counts)),
        )
        ctx["target_timesteps"] = (
            list(video_target.to(torch.float32).split(chunk_counts)),
            list(audio_target.to(torch.float32).split(chunk_counts)),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, clean_latents, start_timesteps, rng. Writes:
        noisy_latents, context_eps."""
        inputs = ctx["inputs"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_timesteps, audio_timesteps = ctx["start_timesteps"]
        video_noises, audio_noises = self._sample_noises(inputs, ctx["rng"])

        ctx["noisy_latents"] = (
            [
                self._noise_chunks(clean, noise, timesteps, layout, "video")
                for clean, noise, timesteps, layout in zip(
                    clean_video, video_noises, video_timesteps, inputs.layouts
                )
            ],
            [
                self._noise_chunks(clean, noise, timesteps, layout, "audio")
                for clean, noise, timesteps, layout in zip(
                    clean_audio, audio_noises, audio_timesteps, inputs.layouts
                )
            ],
        )
        ctx["context_eps"] = self._sample_noises(inputs, ctx["rng"])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def student_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, noisy_latents, clean_latents, context_eps,
        start_timesteps. Writes: student_pred."""
        video_xts, audio_xts = ctx["noisy_latents"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_eps, audio_eps = ctx["context_eps"]
        video_t, audio_t = ctx["start_timesteps"]
        ctx["student_pred"] = self._causal_packed_forward(
            ctx["models"]["backbone"],
            ctx["inputs"],
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=clean_video,
            audio_context=clean_audio,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
        )
        return ctx

    def _step_chunks(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        target_timesteps: torch.Tensor,
        layout: Any,
        modality: str,
        sampler: Any,
    ) -> torch.Tensor:
        """Step one sample's chunks independently and reassemble its time axis."""
        return torch.cat(
            [
                sampler.step_to(
                    pred=pred[self._chunk_slice(chunk, modality)],
                    x_t=x_t[self._chunk_slice(chunk, modality)],
                    t=timestep.unsqueeze(0),
                    s=target_timestep.unsqueeze(0),
                )
                for chunk, timestep, target_timestep in zip(
                    layout.chunks, timesteps, target_timesteps, strict=True
                )
            ],
            dim=1 if modality == "video" else 2,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def solver_step(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, noisy_latents, clean_latents, context_eps,
        start_timesteps, step_timesteps. Writes: solver_xts."""
        inputs = ctx["inputs"]
        video_xts, audio_xts = ctx["noisy_latents"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_eps, audio_eps = ctx["context_eps"]
        video_t, audio_t = ctx["start_timesteps"]
        video_step_t, audio_step_t = ctx["step_timesteps"]

        video_teacher, audio_teacher = self._causal_packed_forward(
            ctx["models"]["tea_model"],
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=clean_video,
            audio_context=clean_audio,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
        )
        ctx["solver_xts"] = (
            [
                self._step_chunks(
                    pred, x_t, timesteps, step_timesteps, layout, "video", self.tea_sampler
                )
                for pred, x_t, timesteps, step_timesteps, layout in zip(
                    video_teacher, video_xts, video_t, video_step_t, inputs.layouts
                )
            ],
            [
                self._step_chunks(
                    pred, x_t, timesteps, step_timesteps, layout, "audio", self.tea_sampler
                )
                for pred, x_t, timesteps, step_timesteps, layout in zip(
                    audio_teacher, audio_xts, audio_t, audio_step_t, inputs.layouts
                )
            ],
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def target_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, solver_xts, clean_latents, context_eps,
        step_timesteps. Writes: target_pred."""
        video_xts, audio_xts = ctx["solver_xts"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_eps, audio_eps = ctx["context_eps"]
        video_t, audio_t = ctx["step_timesteps"]
        ctx["target_pred"] = self._causal_packed_forward(
            ctx["models"]["backbone_ema"],
            ctx["inputs"],
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=clean_video,
            audio_context=clean_audio,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
        )
        return ctx

    def _project(
        self,
        preds: list[torch.Tensor],
        xts: list[torch.Tensor],
        timesteps: list[torch.Tensor],
        targets: list[torch.Tensor],
        inputs: Any,
        modality: str,
    ) -> list[torch.Tensor]:
        """Carry every sample's x0 predictions to the comparison time."""
        return [
            self._step_chunks(pred, x_t, t, s, layout, modality, self.sampler)
            for pred, x_t, t, s, layout in zip(
                preds, xts, timesteps, targets, inputs.layouts, strict=True
            )
        ]

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, student_pred, target_pred, noisy_latents, solver_xts,
        start_timesteps, step_timesteps, target_timesteps. Writes: loss."""
        inputs = ctx["inputs"]
        video_student, audio_student = ctx["student_pred"]
        video_target, audio_target = ctx["target_pred"]

        if self.loss_type == "x_t":
            # Each side travels to the shared comparison time from its own
            # state: the student from the latent it was handed, the target from
            # the solver's output one grid point along. Both arrive at
            # target_timesteps, which is what makes them comparable at all.
            video_cmp, audio_cmp = ctx["target_timesteps"]
            video_student = self._project(
                video_student, ctx["noisy_latents"][0],
                ctx["start_timesteps"][0], video_cmp, inputs, "video",
            )
            audio_student = self._project(
                audio_student, ctx["noisy_latents"][1],
                ctx["start_timesteps"][1], audio_cmp, inputs, "audio",
            )
            video_target = self._project(
                video_target, ctx["solver_xts"][0],
                ctx["step_timesteps"][0], video_cmp, inputs, "video",
            )
            audio_target = self._project(
                audio_target, ctx["solver_xts"][1],
                ctx["step_timesteps"][1], audio_cmp, inputs, "audio",
            )

        # _loss_fn detaches the targets; target_forward already ran no-grad.
        ctx["loss"] = self._loss_fn(
            video_student,
            audio_student,
            inputs=inputs,
            video_targets=video_target,
            audio_targets=audio_target,
            key_prefix="tscd_losses",
        )
        return ctx

    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Roll out on the validation grid rather than the CD training grid.

        Swapping the attributes is what lets the inherited rollout stay
        untouched: ``_rollout_latents`` is shared with DMD and TF and reads
        these three by name. Restored in ``finally`` because the very next
        training step reads them again -- leaking the validation grid into
        ``sample_segment`` would change which timesteps the student is
        distilled on, and nothing downstream would look wrong.
        """
        saved = (self.sampling_timesteps, self.audio_sampling_timesteps, self.sampler)
        self.sampling_timesteps = self.validation_diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = self.validation_diffusion["audio_sampling_timesteps"]
        self.sampler = self.validation_diffusion["sampler"]
        try:
            return super().validate(ctx)
        finally:
            self.sampling_timesteps, self.audio_sampling_timesteps, self.sampler = saved


__all__ = ["CausalMiniMaxH3TSCD"]
