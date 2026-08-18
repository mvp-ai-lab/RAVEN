"""Stateful causal dataset over a precomputed MiniMax H3 latent corpus."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import get_worker_info

from common.data import WorkerResumeContext, WorkerStateEnvelope
from common.seed import yield_seed
from projects.minimax_h3.data.causal_text_only import CausalTextOnlyT2AVDataset


class CausalLatentT2AVDataset(CausalTextOnlyT2AVDataset):
    """Infinite worker-local stream backed by a precomputed MiniMax H3 corpus.

    Prompts come from the ``.pt`` entries themselves; this subclass never reads
    the parent's ``prompts`` field.

    Latents are already in normalized space -- the sampling side applied the
    shift/scale -- so consumers take ``video_latents`` / ``audio_latents`` as
    x0 directly. A corpus of raw VAE output would pass every check here and
    train against the wrong distribution.

    Resume replays the stream from ``(offset, avg_seqlen, cnt)``, which only
    reproduces the same clips while ``latent_paths`` is stable -- adding,
    removing or moving one ``prompt*.pt`` silently remaps every offset onto a
    different file. Nothing but the corpus may be written under ``latent_dir``.
    """

    _STATE_SCHEMA = "jarvis_minimax_h3_causal_latent_worker"
    _STATE_VERSION = 1

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        latent_dir: str,
        **kwargs: Any,
    ) -> None:
        self.latent_paths = sorted(Path(latent_dir).glob("**/prompt*.pt"))
        if not self.latent_paths:
            raise FileNotFoundError(f"No latent files found in: {latent_dir}")

        super().__init__(seed, resume_context, **kwargs)

    def __iter__(self):
        # Structurally identical to the parent's, except for where a sample
        # comes from. Overriding _pack_sample instead is not available: the DMD
        # meta model's _validation_packer instantiates the configured training
        # dataset as a layout packer and calls _pack_sample with real prompt
        # text, so reinterpreting that argument would break validation.
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            offset, avg_seqlen, cnt = self._decoded_worker_states[logical_worker_id]
        else:
            offset, avg_seqlen, cnt = self._initial_worker_state(logical_worker_id)

        while True:
            rng = random.Random(offset)
            samples: list[dict[str, Any]] = []
            cur_seqlen = 0
            num_retries = 0
            while len(samples) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
                path = self.latent_paths[rng.randrange(len(self.latent_paths))]
                entry = torch.load(path, map_location="cpu", weights_only=True)
                # The layout the packer builds is derived from height/width/
                # num_frames, the tensors come off disk, and nothing downstream
                # compares them -- a mismatch surfaces as a reshape error deep
                # inside the model. The inherited width default (1280, so
                # latent_w 80) is already wrong for a 1376-wide corpus, so a
                # config that forgets the override lands here.
                if (
                    entry["video"].shape != self.latent_shape
                    or entry["audio"].shape != self.audio_shape
                ):
                    raise ValueError(
                        f"{path}: corpus latents {tuple(entry['video'].shape)} / "
                        f"{tuple(entry['audio'].shape)} disagree with the configured "
                        f"layout {self.latent_shape} / {self.audio_shape}"
                    )
                candidate = self._pack_sample(prompt=entry["prompt"])
                candidate["video_latents"] = entry["video"]
                candidate["audio_latents"] = entry["audio"]
                candidate_seqlen = candidate["seqlens"]
                if (
                    (
                        self.max_seqlen_per_sample is not None
                        and candidate_seqlen > self.max_seqlen_per_sample
                    )
                    or cur_seqlen + candidate_seqlen > self.max_seqlen
                ):
                    if cur_seqlen + candidate_seqlen > self.max_seqlen:
                        num_retries += 1
                        if num_retries >= self.max_retries:
                            break
                    continue
                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + candidate_seqlen / (cnt + 1)
                cnt += 1
                cur_seqlen += candidate_seqlen
                num_retries = 0
                samples.append(candidate)

            offset = yield_seed(offset)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(
                logical_worker_id, offset, avg_seqlen, cnt
            )
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)


__all__ = ["CausalLatentT2AVDataset"]
