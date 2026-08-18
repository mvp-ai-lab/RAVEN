"""Stateful synthetic causal text-only dataset for DMD training.

The worker-local iterator follows a resumable dynamic packing state
machine and emits immutable post-batch snapshots alongside training batches.

Constructor parameters that used to be injected by
``CausalWan2_1_T2V.setup_dataloader`` are explicit here:

* ``separated_first_frame``: whether the first latent frame is packed alone.
* ``patch_size``: temporal/height/width patch size used for token counts.
* ``vae_stride``: temporal/height/width VAE stride used for latent shape.
"""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from common.data import WorkerResumeContext, WorkerStateEnvelope, WorkerStateLoadError
from common.seed import combine_seed, yield_seed

from utils.flex_attn import _prepare_flex_attention_mask


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _as_int_tuple(name: str, value: Sequence[int], length: int) -> tuple[int, ...]:
    value_tuple = tuple(int(v) for v in value)
    if len(value_tuple) != length or any(v <= 0 for v in value_tuple):
        raise ValueError(f"{name} must contain {length} positive values, got {value_tuple}")
    return value_tuple


class CausalTextOnlyDataset(IterableDataset):
    """Stateful worker-local synthetic causal text-only iterable dataset."""

    _STATE_SCHEMA = "jarvis_causal_text_only_worker"
    _STATE_VERSION = 1

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        paths: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
        default_prompt: str = "A synthetic training video.",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        chunk_size: int | Sequence[int] = 3,
        independent_first_chunk: int | Sequence[int] | None = 3,
        sink: int | Sequence[int] = 0,
        window_size: int | Sequence[int | None] | None = None,
        separated_first_frame: bool = True,
        patch_size: Sequence[int] = (1, 2, 2),
        vae_stride: Sequence[int] = (4, 8, 8),
        z_dim: int = 16,
        video_shape_only: bool = False,
        synthetic_prompt_embeddings: bool = False,
        prompt_embedding_dim: int | None = None,
        prompt_sequence_length: int | None = None,
        total_seqlen: int | None = None,
        max_seqlen: int | None = None,
        max_seqlen_per_sample: int | None = None,
        max_retries: int = 5,
        sync_group_size: int = 1,
    ):
        super().__init__()
        self.seed = int(seed)
        self.resume_context = resume_context
        self.sync_group_size = int(sync_group_size)
        if self.sync_group_size <= 0:
            raise ValueError(f"sync_group_size must be positive, got {sync_group_size}")
        self._decoded_worker_states = {}
        for logical_id, snapshot in resume_context.committed_states.items():
            try:
                state = pickle.loads(snapshot)
                restored = (state["offset"], float(state["avg_seqlen"]), state["cnt"])
            except (KeyError, TypeError, ValueError, pickle.UnpicklingError, EOFError) as exc:
                raise WorkerStateLoadError(f"invalid snapshot for logical worker {logical_id}") from exc
            if (
                (state.get("schema"), state.get("version"), state.get("logical_worker_id"))
                != (self._STATE_SCHEMA, self._STATE_VERSION, logical_id)
                or not isinstance(restored[0], int)
                or not math.isfinite(restored[1])
                or not isinstance(restored[2], int)
                or restored[2] < 0
            ):
                raise WorkerStateLoadError(f"invalid snapshot for logical worker {logical_id}")
            self._decoded_worker_states[logical_id] = restored

        loaded: list[str] = []
        seen: set[str] = set()

        def add_prompt(prompt: str) -> None:
            prompt = prompt.strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                loaded.append(prompt)

        for prompt in prompts or []:
            add_prompt(str(prompt))
        text_files: list[Path] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {raw_path}")
            if path.is_dir():
                text_files.extend(sorted(p for p in path.rglob("*.txt") if p.is_file()))
            elif path.suffix == ".txt":
                text_files.append(path)
            else:
                raise ValueError(f"Only .txt files are supported, got: {raw_path}")
        for file_path in sorted(set(text_files)):
            with file_path.open("r", encoding="utf8") as handle:
                for line in handle:
                    add_prompt(line)
        if not loaded:
            add_prompt(default_prompt)
        if not loaded:
            raise ValueError("No valid prompts found; pass prompts, paths, or default_prompt")
        self.prompts = loaded

        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.z_dim = int(z_dim)
        if min(self.height, self.width, self.num_frames, self.z_dim) <= 0:
            raise ValueError(
                "height, width, num_frames, and z_dim must be positive, "
                f"got {(self.height, self.width, self.num_frames, self.z_dim)}"
            )
        self.separated_first_frame = bool(separated_first_frame)
        self.patch_size = _as_int_tuple("patch_size", patch_size, 3)
        self.vae_stride = _as_int_tuple("vae_stride", vae_stride, 3)
        self.video_shape_only = bool(video_shape_only)
        self.video_shape = (3, self.num_frames, self.height, self.width)

        stride_t, stride_h, stride_w = self.vae_stride
        patch_t, patch_h, patch_w = self.patch_size
        temporal_source = self.num_frames - int(self.separated_first_frame)
        latent_frames = temporal_source // stride_t + int(self.separated_first_frame)
        if temporal_source % stride_t or (latent_frames - int(self.separated_first_frame)) % patch_t:
            raise ValueError("video length is incompatible with VAE stride and temporal patch size")
        if self.height % stride_h or self.width % stride_w:
            raise ValueError("video dimensions must be divisible by the VAE spatial stride")
        latent_height, latent_width = self.height // stride_h, self.width // stride_w
        if latent_height % patch_h or latent_width % patch_w:
            raise ValueError("latent dimensions must be divisible by the spatial patch size")
        self.latent_shape = self.z_dim, latent_frames, latent_height, latent_width

        self.synthetic_prompt_embeddings = bool(synthetic_prompt_embeddings)
        self.prompt_embedding_dim = None if prompt_embedding_dim is None else int(prompt_embedding_dim)
        self.prompt_sequence_length = None if prompt_sequence_length is None else int(prompt_sequence_length)
        if self.synthetic_prompt_embeddings and (
            self.prompt_embedding_dim is None or self.prompt_embedding_dim <= 0
            or self.prompt_sequence_length is None or self.prompt_sequence_length <= 0
        ):
            raise ValueError("prompt embedding dimensions must be positive")
        resolved_max_seqlen = (
            max_seqlen
            if total_seqlen is None
            else int(total_seqlen) // resume_context.world_size
        )
        if resolved_max_seqlen is None or int(resolved_max_seqlen) <= 0:
            raise ValueError(f"max_seqlen must be provided and positive, got {resolved_max_seqlen}")
        self.max_seqlen = int(resolved_max_seqlen)
        self.max_seqlen_per_sample = max_seqlen_per_sample
        self.max_retries = int(max_retries)
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}")

        self.chunk_size = [int(x) for x in _as_list(chunk_size)]
        self.independent_first_chunk = _as_list(independent_first_chunk)
        self.sink = [int(x) for x in _as_list(sink)]
        self.window_size = _as_list(window_size)
        if not (
            len(self.chunk_size)
            == len(self.independent_first_chunk)
            == len(self.sink)
            == len(self.window_size)
        ):
            raise ValueError("chunk_size, independent_first_chunk, sink and window_size must have the same length")
        self.independent_first_chunk = [
            int(cs if ifc is None else ifc) for ifc, cs in zip(self.independent_first_chunk, self.chunk_size)
        ]
        self.window_size = [None if value is None else int(value) for value in self.window_size]
        for index, (chunk_size, first_chunk) in enumerate(zip(self.chunk_size, self.independent_first_chunk)):
            remaining = latent_frames - first_chunk
            if not (chunk_size > 0 and 0 < first_chunk <= latent_frames and remaining % chunk_size == 0):
                raise ValueError(
                    f"invalid causal layout {index}: chunk_size={chunk_size}, "
                    f"independent_first_chunk={first_chunk}, latent_frames={latent_frames}"
                )

    @staticmethod
    def _make_generator(seed: int) -> torch.Generator:
        return torch.Generator().manual_seed(seed)

    def _initial_worker_state(self, logical_worker_id: int) -> tuple[int, float, int]:
        offset = (
            self.seed
            + (self.resume_context.rank // self.sync_group_size)
            * (self.resume_context.num_workers or 1)
            + logical_worker_id
        )
        return offset, 0.0, 0

    def _encode_worker_state(
        self, logical_worker_id: int, offset: int, avg_seqlen: float, cnt: int
    ) -> bytes:
        return pickle.dumps({
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "logical_worker_id": logical_worker_id,
            "offset": offset,
            "avg_seqlen": avg_seqlen,
            "cnt": cnt,
        })

    def initial_worker_snapshot(self, logical_worker_id: int) -> bytes:
        return self._encode_worker_state(logical_worker_id, *self._initial_worker_state(logical_worker_id))

    def __iter__(self):
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                f"worker topology mismatch: context expects {effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            offset, avg_seqlen, cnt = self._decoded_worker_states[logical_worker_id]
        else:
            offset, avg_seqlen, cnt = self._initial_worker_state(logical_worker_id)
        worker_prompts = list(self.prompts)
        random.Random(self.seed).shuffle(worker_prompts)

        while True:
            rng = random.Random(offset)
            samples: list[dict[str, Any]] = []
            cur_seqlen = 0
            num_retries = 0
            attempt_index = 0
            while len(samples) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
                prompt = worker_prompts[rng.randrange(len(worker_prompts))]
                candidate = self._build_one_sample(
                    prompt=prompt, rng=rng, offset=offset, attempt_index=attempt_index
                )
                attempt_index += 1
                candidate_seqlen = candidate["seqlens"] if candidate is not None else 0
                if (
                    candidate is None
                    or (
                        self.max_seqlen_per_sample is not None
                        and candidate_seqlen > self.max_seqlen_per_sample
                    )
                    or cur_seqlen + candidate_seqlen > self.max_seqlen
                ):
                    if candidate is not None and cur_seqlen + candidate_seqlen > self.max_seqlen:
                        num_retries += 1
                        if num_retries >= self.max_retries:
                            break
                    continue
                assert candidate is not None
                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + candidate_seqlen / (cnt + 1)
                cnt += 1
                cur_seqlen += candidate_seqlen
                num_retries = 0
                samples.append(candidate)

            offset = yield_seed(offset)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(logical_worker_id, offset, avg_seqlen, cnt)
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)

    def _build_one_sample(
        self, *, prompt: str, rng: random.Random, offset: int, attempt_index: int
    ) -> dict[str, Any] | None:
        causal_idx = rng.randint(0, len(self.chunk_size) - 1)
        chunk_size = self.chunk_size[causal_idx]
        independent_first_chunk = self.independent_first_chunk[causal_idx]
        sink = self.sink[causal_idx]
        window_size = self.window_size[causal_idx]

        sample = self._pack_sample(
            prompt=prompt,
            chunk_size=chunk_size,
            independent_first_chunk=independent_first_chunk,
            sink=sink,
            window_size=window_size,
        )
        if self.video_shape_only:
            sample["video_shapes"] = self.video_shape
            sample["latent_shapes"] = self.latent_shape
        else:
            video_generator = self._make_generator(combine_seed(offset, attempt_index, "video"))
            sample["videos"] = torch.rand(
                self.video_shape,
                dtype=torch.float32,
                generator=video_generator,
            )
        if self.synthetic_prompt_embeddings:
            prompt_embedding_generator = self._make_generator(
                combine_seed(offset, attempt_index, "prompt_embedding")
            )
            negative_embedding_generator = self._make_generator(
                combine_seed(offset, attempt_index, "negative_prompt_embedding")
            )
            embedding_shape = (self.prompt_sequence_length, self.prompt_embedding_dim)
            sample["prompt_embs"] = torch.randn(
                embedding_shape,
                dtype=torch.float32,
                generator=prompt_embedding_generator,
            )
            sample["neg_prompt_embs"] = torch.randn(
                embedding_shape,
                dtype=torch.float32,
                generator=negative_embedding_generator,
            )
        return sample

    def _pack_sample(
        self,
        *,
        prompt: str,
        chunk_size: int,
        independent_first_chunk: int,
        sink: int,
        window_size: int | None,
    ) -> dict:
        separated_first_frame = int(self.separated_first_frame)
        _, lat_t, lat_h, lat_w = self.latent_shape
        patch_t, patch_h, patch_w = self.patch_size
        seqlen_per_frame = (lat_h // patch_h) * (lat_w // patch_w)
        if self.separated_first_frame:
            remaining_latent_frames = lat_t - 1
            temporal_tokens = 1 + remaining_latent_frames // patch_t
        else:
            temporal_tokens = lat_t // patch_t
        seqlen = temporal_tokens * seqlen_per_frame
        seqlen_per_chunk = chunk_size * seqlen_per_frame
        seqlen_first_chunk = seqlen_per_frame * independent_first_chunk
        num_chunks = 1 + (lat_t - independent_first_chunk) // chunk_size

        position_id: list[int] = []
        latent_index: list[int] = []
        latent_seqlen: list[int] = []
        noisy_latent_relative_index: list[int] = []
        noisy_latent_seqlen: list[int] = []
        split_lens: list[int] = []
        attn_modes: list[str] = []
        frame_shifts: list[int] = []
        sample_len = 0
        curr, curr_noisy, curr_rope, curr_frame = 0, 0, 0, 0

        for j in range(num_chunks):
            if j == 0 and independent_first_chunk:
                position_id.extend([curr_rope] * seqlen_first_chunk)
                latent_index.extend(range(curr, curr + seqlen_first_chunk))
                noisy_latent_relative_index.extend(range(curr_noisy, curr_noisy + seqlen_first_chunk))
                curr += seqlen_first_chunk
                curr_noisy += seqlen_first_chunk
                latent_seqlen.append(seqlen_first_chunk)
                noisy_latent_seqlen.append(seqlen_first_chunk)
                split_lens.append(seqlen_first_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_first_chunk

                position_id.extend([curr_rope] * seqlen_first_chunk)
                latent_index.extend(range(curr, curr + seqlen_first_chunk))
                curr += seqlen_first_chunk
                curr_noisy += seqlen_first_chunk
                latent_seqlen.append(seqlen_first_chunk)
                split_lens.append(seqlen_first_chunk)
                attn_modes.append("full")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_first_chunk

                curr_rope += 1
                curr_frame += independent_first_chunk

            elif j == num_chunks - 1:
                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(range(curr, curr + seqlen_per_chunk))
                noisy_latent_relative_index.extend(range(curr_noisy, curr_noisy + seqlen_per_chunk))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                noisy_latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                curr_rope += 1
                curr_frame += chunk_size

            else:
                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(range(curr, curr + seqlen_per_chunk))
                noisy_latent_relative_index.extend(range(curr_noisy, curr_noisy + seqlen_per_chunk))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                noisy_latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(range(curr, curr + seqlen_per_chunk))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("full")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                curr_rope += 1
                curr_frame += chunk_size

        q_ranges, k_ranges, attn_type_map, attn_workloads = _prepare_flex_attention_mask(
            split_lens,
            attn_modes,
            sink=sink,
            window_size=window_size,
        )

        return dict(
            prompts=prompt,
            seqlens=seqlen,
            position_ids=torch.tensor(position_id, dtype=torch.int32),
            latent_indexes=torch.tensor(latent_index, dtype=torch.int32),
            latent_seqlens=torch.tensor(latent_seqlen, dtype=torch.int32),
            noisy_latent_relative_indexes=torch.tensor(noisy_latent_relative_index, dtype=torch.int32),
            noisy_latent_seqlens=torch.tensor(noisy_latent_seqlen, dtype=torch.int32),
            split_lens=split_lens,
            attn_modes=attn_modes,
            attn_workloads=attn_workloads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sample_lens=sample_len,
            frame_shifts=frame_shifts,
            chunk_sizes=chunk_size,
            independent_first_chunks=independent_first_chunk,
            sinks=sink,
            window_sizes=window_size,
        )
