# SPDX-License-Identifier: Apache-2.0
"""Training timesteps that emit a paired (video, audio) draw for MiniMax-H3.

H3 advances ONE step index through two grids that carry different shifts, so
along its own trajectory ``sigma_video != sigma_audio`` everywhere except the
endpoints. Training has to land on those same pairs. Two independent draws miss
them, and so does a single shared sigma -- the shared sigma is just as far off
the curve as the independent draw is. What lands on it is one draw from the base
distribution pushed through two shifts, which is what ``sample_pair`` does.

Owning the pair here rather than in a meta model is what makes the invariant
structural. A two-node config can only state "these differ in nothing but shift"
by convention, and a convention is exactly what a later edit breaks silently.
Here the audio side IS the video object with one field replaced, so there is no
second set of distribution parameters that can drift.

The ``_sample_raw`` bodies restate the draw from the matching class in
``common/diffusion/timestep/training``. That duplication is deliberate: the
distribution lives inside ``sample()``, which returns an already-shifted value,
and a shifted value cannot be un-shifted once ``postprocess_sample`` has clamped
it. Keep these two lines in step with common if the base classes ever change.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch

from common.diffusion.timestep.training.logitnormal import LogitNormalTrainingTimesteps
from common.diffusion.timestep.training.uniform import UniformTrainingTimesteps


class _PairedTimestepsMixin:
    """Adds an audio twin that differs from ``self`` in ``shift`` alone."""

    def __init__(self, *, audio_shift: float, **kwargs) -> None:
        super().__init__(**kwargs)
        # dynamic_shift derives the shift from seqlen inside postprocess_sample
        # and ignores self.shift, so the twin would silently be handed the video
        # shift and the pair would collapse onto one sigma.
        assert not self.dynamic_shift, (
            "paired timesteps require a static shift: dynamic_shift computes the "
            "shift from seqlen and would give video and audio the same one"
        )
        # A shallow copy, not a second construction: everything except shift --
        # T, scaling, clamps, and the distribution object itself -- is shared by
        # identity rather than by matching arguments.
        self._audio = copy.copy(self)
        self._audio.shift = float(audio_shift)

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    def sample_pair(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draw from the base distribution, shifted two ways.

        The caller owns the RNG: seed the global stream around this call the way
        the rest of the phase does, exactly as it would for ``sample``.
        """
        raw = self._sample_raw(size)
        return (
            self.postprocess_sample(raw.clone(), seqlens, device),
            self._audio.postprocess_sample(raw.clone(), seqlens, device),
        )


class PairedUniformTrainingTimesteps(_PairedTimestepsMixin, UniformTrainingTimesteps):
    """Uniform on [0, 1), then each modality's shift."""

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        return torch.rand(size, dtype=torch.float64)


class PairedLogitNormalTrainingTimesteps(_PairedTimestepsMixin, LogitNormalTrainingTimesteps):
    """sigmoid(N(loc, scale)), then each modality's shift."""

    def _sample_raw(self, size: Sequence[int]) -> torch.Tensor:
        return self.dist.sample(size)[..., 0].to(torch.float64)


__all__ = [
    "PairedUniformTrainingTimesteps",
    "PairedLogitNormalTrainingTimesteps",
]
