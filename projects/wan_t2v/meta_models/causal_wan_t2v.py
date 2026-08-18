"""CausalWan
T2V: causal Wan2.1 T2V machinery shared by the DMD and GRPO algorithms.

This base class holds only what is byte-identical between
:class:`~projects.wan_t2v.meta_models.causal_wan_t2v_dmd.CausalWanT2VDMD` and
:class:`~projects.wan_t2v.meta_models.causal_wan_t2v_grpo.CausalWanT2VGRPO`:
the ``ForwardInput`` payload, the validation spec, input preparation, the
packed/flat forward wrappers and the small helpers around them. Anything that
differs between the two algorithms (``rollout``, ``validate``, ``_infer``,
``_loss_fn``, ...) lives in the subclasses.

It is never named in a config -- it exists only as a Python base.

``__init__`` reads the ``meta_model`` keys both algorithms share and the three
``config.diffusion`` components both build (``sampling_timesteps``, ``schedule``,
``sampler``); the full component dict is kept on ``self._diffusion`` so a
subclass can pick up its extra components from the same instantiation.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from common import media
from common.diffusion import build_diffusion
from common.distributed import ops
from common.distributed.ops import all_reduce_max, all_reduce_sum, get_device, get_world_size
from common.distributed.unified_parallel import (
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from common.logging import get_logger
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed

from projects.wan_t2v.modeling.causal_model import NaiveCache
from utils.flex_attn import create_sparse_mask
from torch.distributed.fsdp import FSDPModule

logger = get_logger()

# RAVEN: causal_wan2_1_t2v.py:39
FLEX_ATTN_BLOCK_SIZE = 128


def _read_video_frames(path: str | Path) -> torch.Tensor:
    """Read all MP4 video frames as a ``uint8`` THWC tensor using PyAV."""
    import av

    with av.open(str(path)) as container:
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        ]
    return torch.stack(frames)

def _resize_video_grid(videos: torch.Tensor, nrow: int) -> torch.Tensor:
    """Scale a padded ``[B,C,T,H,W]`` grid batch to a bounded pixel budget."""
    cols = nrow
    rows = math.ceil(videos.shape[0] / cols)
    max_pixels = 1920 * 1080 // (cols * rows)
    height, width = videos.shape[-2:]
    if height * width <= max_pixels:
        return videos

    scale_factor = (max_pixels / (height * width)) ** 0.5
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    new_height -= new_height % 2
    new_width -= new_width % 2
    batch, channels, frames = videos.shape[:3]
    resized = videos.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    resized = F.interpolate(resized, size=(new_height, new_width), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, new_height, new_width).permute(0, 2, 1, 3, 4)

def _stack_padded_videos(videos: list[torch.Tensor]) -> torch.Tensor:
    """Pad ``[C,T,H,W]`` videos to common dimensions and stack a batch."""
    max_t = max(video.shape[1] for video in videos)
    max_h = max(video.shape[2] for video in videos)
    max_w = max(video.shape[3] for video in videos)
    padded = []
    for video in videos:
        if video.shape[1] < max_t:
            # Hold the final frame so shorter clips do not end with a black flash.
            video = torch.cat(
                [video, video[:, -1:].expand(-1, max_t - video.shape[1], -1, -1)],
                dim=1,
            )
        pad_w = max_w - video.shape[3]
        pad_h = max_h - video.shape[2]
        video = F.pad(
            video,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=-1.0,
        )
        padded.append(video)
    return torch.stack(padded)

def _bin_losses(
    losses: torch.Tensor,
    seqlens: torch.Tensor,
    values: torch.Tensor,
    bin_size: int | float = 100,
    min_value: int | float = 0,
    max_value: int | float = 1000,
    key_prefix: str = "values",
) -> tuple[dict[str, float], dict[str, float]]:
    """Bucket per-sample losses by ``values`` and average per token within bins."""
    # RAVEN: project/utils/misc.py:235-273 (bin_losses)
    nbins = round((max_value - min_value) / bin_size)
    bin_size = (max_value - min_value) / nbins  # recompute to avoid rounding drift

    if not (values.min() >= min_value and values.max() <= max_value):
        raise ValueError(f"Values out of range [{min_value}, {max_value}]")

    bin_indices = ((values - min_value) / bin_size).long().clamp(0, nbins - 1)
    binned_losses = torch.zeros((nbins,), dtype=losses.dtype, device=losses.device)
    binned_seqlens = torch.zeros((nbins,), dtype=seqlens.dtype, device=seqlens.device)
    binned_losses.scatter_add_(0, bin_indices, losses)
    binned_seqlens.scatter_add_(0, bin_indices, seqlens)

    binned_loss_dict: dict[str, float] = {}
    binned_cnt_dict: dict[str, float] = {}
    for bin_index in range(nbins):
        if binned_seqlens[bin_index] <= 0:
            continue
        bin_start = min_value + bin_index * bin_size
        bin_end = bin_start + bin_size
        if isinstance(bin_size, float):
            bin_key = f"{key_prefix}/{bin_start:.3f}-{bin_end:.3f}"
        else:
            bin_key = f"{key_prefix}/{int(bin_start)}-{int(bin_end)}"
        binned_loss_dict[bin_key] = (binned_losses[bin_index] / binned_seqlens[bin_index]).item()
        binned_cnt_dict[bin_key] = binned_seqlens[bin_index].item()

    return binned_loss_dict, binned_cnt_dict


# Merged from RAVEN's three-level hierarchy into ONE dataclass:
#   BaseForwardInput            (RAVEN: project/meta_models/base_meta_model.py:27-34)
#   Wan2_1_T2VForwardInput      (RAVEN: project/meta_models/wan2_1_t2v.py:33-36)
#   CausalWan2_1_T2VForwardInput(RAVEN: project/meta_models/causal_wan2_1_t2v.py:47-74)
# Pruned fields (not used on our T2V/DMD path):
#   - text-token packing plumbing (packed_text_tokens / packed_text_indexes,
#     add_bos / pad handling) — prompts are encoded to context directly.
#   - i2v/CLIP conditioning: past_key_values_cross_attn_img,
#     update_past_key_values_cross_attn_img.
# KEPT: inference self/cross attn KV caches — rollout needs them.
@dataclass
class ForwardInput:
    # RAVEN: base_meta_model.py:27-34
    xts: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    timesteps: torch.Tensor | None = None
    latents: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    noises: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    batch_size: torch.Tensor | None = None
    seqlens: torch.Tensor | None = None
    prompts: list[str] | None = None  # for reward function only, not used in forward pass
    # RAVEN: wan2_1_t2v.py:33-36
    context: list[torch.Tensor] | None = None
    max_seq_len: int | None = None
    guidance: torch.Tensor | None = None
    # RAVEN: causal_wan2_1_t2v.py:47-67
    # common packed inputs prepared in training dataloader and infer func respectively
    packed_position_ids: torch.Tensor | None = None
    packed_latent_indexes: torch.Tensor | None = None
    packed_latent_seqlens: torch.Tensor | None = None
    packed_noisy_latent_relative_indexes: torch.Tensor | None = None
    packed_noisy_latent_seqlens: torch.Tensor | None = None
    sample_lens: list[int] | None = None
    frame_shifts: list[list[int]] | None = None  # per-sample lists; flattened in _pred
    # universal inputs prepared in training dataloader and prepare_inference_inputs respectively
    chunk_sizes: list[int] | None = None
    independent_first_chunks: list[int] | None = None
    sinks: list[int] | None = None
    window_sizes: list[int | None] | None = None
    # training only, prepared in training dataloader
    split_lens: list[int] | None = None
    attn_modes: list[str] | None = None
    q_ranges: torch.Tensor | None = None
    k_ranges: torch.Tensor | None = None
    attn_type_map: torch.Tensor | None = None
    attn_workloads: list[int] | None = None
    # inference only: initialized within rollout and updated in each sampling step
    # RAVEN: causal_wan2_1_t2v.py:68-72 (i2v img cache pruned)
    past_key_values_self_attn: Any = None
    update_past_key_values_self_attn: bool = False
    past_key_values_cross_attn: Any = None
    update_past_key_values_cross_attn: bool = False


@dataclass(frozen=True)
class _ValidationSpec:
    latent_shape: tuple[int, int, int, int]
    seqlen: int
    chunk_size: int
    first_chunk: int
    sink: int
    window_size: int | None
    guidance_scale: float
    fps: int
    nrow: int
    crf: int


class CausalWanT2V:
    """Shared causal-Wan machinery; see module docstring."""

    def __init__(self, config):
        self.config = config
        mm = config.meta_model
        self.dummy_latents = mm.dummy_latents
        self.guidance_min = mm.guidance_min
        self.guidance_max = mm.guidance_max
        self.default_neg_prompt = mm.default_neg_prompt
        self.chunk_wise_weighting = dict(mm.chunk_wise_weighting) if mm.chunk_wise_weighting else None

        self._diffusion = build_diffusion(config.diffusion)
        self.sampling_timesteps = self._diffusion["sampling_timesteps"]
        self.schedule = self._diffusion["schedule"]
        self.sampler = self._diffusion["sampler"]

        self._neg_context = None  # lazily encoded once; deterministic, hence resume-safe
        self._warned_weighting: set[str] = set()  # RAVEN log_n_times equivalent: warn once per key

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models, rng, config. Writes: inputs, neg_inputs."""
        # RAVEN: project/meta_models/causal_wan2_1_t2v.py:329-425 (prepare_training_inputs)
        batch = ctx["batch"]
        models = ctx["models"]
        rng = ctx["rng"]
        device = get_device()

        # move all tensors to device
        # RAVEN: causal_wan2_1_t2v.py:330-331
        batch = self._to_device(batch, device)

        # Latents. The trial dataset runs in video_shape_only mode: no pixels are
        # shipped, only per-sample latent shapes, so only the dummy path exists.
        # RAVEN: causal_wan2_1_t2v.py:333-347 (dummy_latents branch)
        assert self.dummy_latents, "only dummy_latents=True is implemented (video_shape_only dataset)"
        latent_shapes = batch["latent_shapes"]
        latents = [
            torch.empty(*shape, device=device).normal_(generator=rng.torch_cuda_generator)
            for shape in latent_shapes
        ]
        batch_size_int = len(latents)

        # encode prompts (real-prompt mode: text is always present)
        # RAVEN: causal_wan2_1_t2v.py:349-355
        texts = batch["prompts"]
        assert texts is not None, "batch must provide 'prompts' (real-prompt mode)"
        context = self._encode_prompts(models=models, texts=texts)

        # Guidance is drawn from rng EVEN though guidance_min == guidance_max:
        # skipping the draw would shift the RNG stream and break bit-exact
        # equivalence with RAVEN.
        # RAVEN: causal_wan2_1_t2v.py:357-361
        guidance = (
            torch.rand(batch_size_int, device=device, generator=rng.torch_cuda_generator)
            * (self.guidance_max - self.guidance_min)
            + self.guidance_min
        )

        # pack data: offset per-sample index tensors into the packed coordinate frame
        # RAVEN: causal_wan2_1_t2v.py:363-372
        sample_lens = batch["sample_lens"]
        sample_lens_cumsum = torch.cumsum(torch.tensor([0] + sample_lens), dim=0)[:-1]
        packed_position_ids = torch.cat(batch["position_ids"], dim=0)
        packed_latent_indexes = torch.cat(
            [latent_indexes + sample_lens_cumsum[i] for i, latent_indexes in enumerate(batch["latent_indexes"])],
            dim=0,
        )
        packed_latent_seqlens = torch.cat(batch["latent_seqlens"], dim=0)
        packed_noisy_latent_seqlens = torch.cat(batch["noisy_latent_seqlens"], dim=0)
        q_ranges = torch.cat([q_range + sample_lens_cumsum[i] for i, q_range in enumerate(batch["q_ranges"])])
        k_ranges = torch.cat([k_range + sample_lens_cumsum[i] for i, k_range in enumerate(batch["k_ranges"])])
        attn_type_map = torch.cat(batch["attn_type_map"])

        # RAVEN: causal_wan2_1_t2v.py:374-384
        curr, curr_noisy = 0, 0
        noisy_latent_relative_indexes = list()
        for i in range(len(sample_lens)):
            sample_len = sample_lens[i]
            noisy_latent_relative_index = batch["noisy_latent_relative_indexes"][i]
            sample_indexes = torch.tensor(list(range(curr, curr + sample_len)), dtype=torch.int32, device=device)
            sample_latent_indexes = packed_latent_indexes[torch.isin(packed_latent_indexes, sample_indexes)]
            noisy_latent_relative_indexes.append(curr_noisy + noisy_latent_relative_index)
            curr += sample_len
            curr_noisy += len(sample_latent_indexes)
        packed_noisy_latent_relative_indexes = torch.cat(noisy_latent_relative_indexes, dim=0)

        # construct pos inputs
        # RAVEN: causal_wan2_1_t2v.py:386-418
        max_seq_len = max(batch["seqlens"])
        seqlens = torch.tensor(batch["seqlens"], dtype=torch.int32, device=device)
        batch_size = torch.tensor(batch_size_int, dtype=torch.int32, device=device)
        inputs = ForwardInput(
            latents=latents,
            batch_size=batch_size,
            seqlens=seqlens,
            context=context,
            max_seq_len=max_seq_len,
            guidance=guidance,
            prompts=texts,
            # packed
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
            frame_shifts=batch["frame_shifts"],
            # training only
            split_lens=batch["split_lens"],
            attn_modes=batch["attn_modes"],
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=batch["attn_workloads"],
            # universal
            chunk_sizes=batch["chunk_sizes"],
            independent_first_chunks=batch["independent_first_chunks"],
            sinks=batch["sinks"],
            window_sizes=batch["window_sizes"],
        )

        # Negative inputs: same conditioning except the text context, which is the
        # default negative prompt. Encode it once and cache — the encoding is
        # deterministic, hence resume-safe.
        # RAVEN: causal_wan2_1_t2v.py:420-423 + wan2_1_t2v.py:228-242 (prepare_negative_inputs)
        if self._neg_context is None:
            self._neg_context = self._encode_prompts(models=models, texts=[self.default_neg_prompt])[0]
        # clone-then-replace: sharing tensor refs with `inputs` would break clone
        # discipline (mirrors RAVEN's deepcopy in prepare_negative_inputs)
        neg_inputs = replace(self._clone_inputs(inputs), context=[self._neg_context for _ in range(batch_size_int)])

        ctx["inputs"] = inputs
        ctx["neg_inputs"] = neg_inputs
        return ctx

    @staticmethod
    def _validation_prompts(config: Any) -> list[str]:
        validation = config.validation
        prompts = validation.get("prompts", None)
        if prompts:
            return [str(prompt) for prompt in prompts]
        prompt_file = validation.get("prompt_file", None)
        if prompt_file is None:
            return []
        return [
            line.strip()
            for line in Path(prompt_file).expanduser().read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def _completed_prompt_ids(media_dir: Path, variant_names: list[str]) -> set[int]:
        """Prompt indices already rendered for every variant."""
        completed: set[int] | None = None
        for name in variant_names:
            ids = {
                int(path.name[6:10])
                for path in (media_dir / "validation" / name).glob("prompt*.mp4")
            }
            completed = ids if completed is None else completed & ids
        return completed or set()

    @staticmethod
    def _split_validation_prompts(
        pending: list[tuple[int, str]], rank: int, world_size: int
    ) -> list[dict[str, Any]]:
        """Equal-length per-rank slices; the padding entries are not saved."""
        if not pending:
            return []
        per_rank = (len(pending) + world_size - 1) // world_size
        total = per_rank * world_size
        padded = pending + [pending[-1]] * (total - len(pending))
        start = rank * per_rank
        return [
            {
                "prompt": prompt,
                "prompt_idx": index,
                "should_save": start + offset < len(pending),
            }
            for offset, (index, prompt) in enumerate(padded[start : start + per_rank])
        ]

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """The shared validation loop for every causal-Wan T2V meta model.

        ``validation.batch_size`` (default 1) packs that many prompts into ONE
        rollout forward; each prompt keeps its own deterministic noise and
        sampler streams (``validation.shift_seed``, default true, shifts the
        seeds by prompt index; false pins every prompt to the same seed).
        ``validation.validate_ema`` renders the EMA shadow as its own variant.
        Resume drops prompts already on disk (rank 0's view, all-gathered);
        every rank walks the same entry count so the collectives stay aligned.
        Sequence parallelism is handled when active; without it the split is
        plain per-rank.

        This loop is the DMD path unified for GRPO: a GRPO run picks up
        resume-skipping and the SP split with it (behavior changes from the
        old per-project validate, deliberate).
        """
        config = ctx["config"]
        models = ctx["models"]
        step = ctx["step"]
        validation = config.validation
        prompts = self._validation_prompts(config)
        assert prompts, "validation requires at least one prompt"
        rank = ops.get_rank()

        variants = [("backbone", models["backbone"])]
        if validation.get("validate_ema", False):
            variants.append(("backbone_ema", models["backbone_ema"]))
        media_dir = logger.iter_dir("media", step)

        # Resume: drop prompts already on disk. Every rank adopts rank 0's
        # view -- a divergent view would give ranks different per-rank counts,
        # desynchronise the FSDP all-gathers, and hang.
        completed = ops.all_gather_object(
            self._completed_prompt_ids(media_dir, [name for name, _ in variants])
        )[0]
        pending = [
            (index, prompt)
            for index, prompt in enumerate(prompts)
            if index not in completed
        ]

        sp_enabled = is_unified_parallel_initialized() and get_unified_parallel_world_size() > 1
        if sp_enabled:
            up_size = get_unified_parallel_world_size()
            group_idx = rank // up_size
            num_groups = ops.get_world_size() // up_size
            entries = self._split_validation_prompts(pending, group_idx, num_groups)
            is_group_main = rank % up_size == 0
        else:
            group_idx = rank
            entries = self._split_validation_prompts(pending, rank, ops.get_world_size())
            is_group_main = True

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str, str]] = []
        try:
            for _, model in variants:
                model.eval()

            device = get_device()
            vae = models["vae"]
            spec = self._validation_spec(config, vae, models["backbone"])
            seqlens = torch.tensor([spec.seqlen], dtype=torch.int32, device=device)
            self.sampling_timesteps.set_timesteps(seqlen=seqlens, device=device)

            batch_size = int(validation.get("batch_size", 1))
            assert batch_size >= 1, f"validation.batch_size must be >= 1, got {batch_size}"
            shift_seed = bool(validation.get("shift_seed", True))
            deadline = float(validation.get("max_hours", 0)) * 3600
            started = time.monotonic()
            for position in range(0, len(entries), batch_size):
                batch_entries = entries[position : position + batch_size]
                if deadline > 0:
                    expired = torch.tensor(
                        int(time.monotonic() - started >= deadline),
                        dtype=torch.int32,
                        device="cpu",
                    )
                    ops.all_reduce_max(expired)
                    if expired.item():
                        logger.info(
                            "Validation runtime limit reached after %d of %d local prompts",
                            position, len(entries),
                        )
                        break
                prompts_batch = [entry["prompt"] for entry in batch_entries]
                prompt_idxs = [entry["prompt_idx"] for entry in batch_entries]
                # One encoder call for the whole batch plus the shared negative.
                encoded = self._encode_prompts(
                    models=models,
                    texts=[*prompts_batch, self.default_neg_prompt],
                )
                positives, negative = encoded[:-1], encoded[-1]
                noise_rngs = [
                    RandomState(
                        combine_seed(
                            int(validation.seed), "validation_noise",
                            prompt_idx if shift_seed else 0,
                        )
                    )
                    for prompt_idx in prompt_idxs
                ]
                initial_noises = [
                    torch.empty(
                        spec.latent_shape,
                        device=device,
                        dtype=torch.float32,
                    ).normal_(
                        generator=(
                            rng.torch_cuda_generator
                            if device.type == "cuda"
                            else rng.torch_generator
                        )
                    )
                    for rng in noise_rngs
                ]
                pos_inputs, neg_inputs = self._build_validation_inputs(
                    prompts=prompts_batch,
                    positives=positives,
                    negatives=[negative] * len(prompts_batch),
                    initial_noises=initial_noises,
                    spec=spec,
                    device=device,
                )

                sampling_seed = combine_seed(
                    int(validation.seed), "validation_sampler",
                    prompt_idxs[0] if shift_seed else 0,
                )
                # Per-prompt sampler streams inside the batched rollout
                # (samplers without an rng parameter ignore them).
                sampler_rngs = [
                    RandomState(
                        combine_seed(
                            int(validation.seed), "validation_sampler",
                            prompt_idx if shift_seed else 0,
                        )
                    )
                    for prompt_idx in prompt_idxs
                ]
                for variant_name, backbone in variants:
                    final_x0s = self._rollout_validation_latents(
                        backbone=backbone,
                        sampling_seed=sampling_seed,
                        pos_inputs=pos_inputs,
                        neg_inputs=neg_inputs,
                        rngs=sampler_rngs,
                    )
                    for index, entry in enumerate(batch_entries):
                        should_decode = is_group_main and (
                            not sp_enabled or entry["should_save"]
                        )
                        if should_decode:
                            video = self._decode_validation_latents(vae, [final_x0s[index]])
                        if is_group_main and entry["should_save"]:
                            if sp_enabled:
                                name = (
                                    f"validation/{variant_name}/"
                                    f"prompt{entry['prompt_idx']:04d}_group{group_idx:04d}.mp4"
                                )
                            else:
                                name = (
                                    f"validation/{variant_name}/"
                                    f"prompt{entry['prompt_idx']:04d}_rank{rank:04d}.mp4"
                                )
                            path = media_dir / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            media.save_video(
                                video,
                                path,
                                fps=spec.fps,
                                nrow=spec.nrow,
                                crf=spec.crf,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                            saved_items.append(
                                (variant_name, name, str(path), f"[{variant_name}] {entry['prompt']}")
                            )
                        if should_decode:
                            del video
                    del final_x0s
                del pos_inputs, neg_inputs, encoded, initial_noises, sampler_rngs

            gathered = ops.all_gather_object(saved_items)
            try:
                if rank == 0:
                    all_items = [item for rank_items in gathered for item in rank_items]
                    for variant_name, _ in variants:
                        variant_items = [item for item in all_items if item[0] == variant_name]
                        if not variant_items:
                            continue
                        paths = {name: Path(path) for _, name, path, _ in variant_items}
                        captions = {name: caption for _, name, _, caption in variant_items}
                        logger.log_video(
                            paths,
                            step=step,
                            collection=f"validation/{variant_name}/videos",
                            captions=captions,
                            fps=spec.fps,
                        )
                        if not validation.get("save_grid", True):
                            continue
                        try:
                            videos = []
                            for _, name, _, _ in variant_items:
                                frames = _read_video_frames(paths[name])
                                videos.append(frames.permute(3, 0, 1, 2).float().div(127.5).sub(1.0))
                            grid = _stack_padded_videos(videos)
                            grid = _resize_video_grid(grid, spec.nrow)
                            grid_name = f"validation/{variant_name}/grid.mp4"
                            grid_path = media_dir / grid_name
                            grid_path.parent.mkdir(parents=True, exist_ok=True)
                            media.save_video(
                                grid,
                                grid_path,
                                fps=spec.fps,
                                nrow=spec.nrow,
                                crf=spec.crf,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                            logger.log_video({grid_name: grid_path}, step=step)
                        except Exception as exc:
                            logger.warning("Validation grid build failed: %s", exc)
            finally:
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)
        return ctx

    @staticmethod
    def _build_validation_inputs(
        *,
        prompts: Sequence[str],
        positives: Sequence[torch.Tensor],
        negatives: Sequence[torch.Tensor],
        initial_noises: Sequence[torch.Tensor],
        spec: _ValidationSpec,
        device: torch.device,
    ) -> tuple[ForwardInput, ForwardInput]:
        """One packed ForwardInput over ``prompts`` (validation batch_size).

        Every per-sample field is the list of its per-prompt values; the
        noise list is what the rollout consumes sample-by-sample, so each
        prompt keeps its own stream. ``max_seq_len`` follows the training
        convention (max over seqlens).
        """
        batch_size = len(prompts)
        common = dict(
            noises=list(initial_noises),
            batch_size=torch.tensor(batch_size, dtype=torch.int32, device=device),
            seqlens=torch.tensor([spec.seqlen] * batch_size, dtype=torch.int32, device=device),
            max_seq_len=spec.seqlen,
            guidance=torch.tensor(
                [spec.guidance_scale] * batch_size, dtype=torch.float32, device=device
            ),
            chunk_sizes=[spec.chunk_size] * batch_size,
            independent_first_chunks=[spec.first_chunk] * batch_size,
            sinks=[spec.sink] * batch_size,
            window_sizes=[spec.window_size] * batch_size,
        )
        pos_inputs = ForwardInput(prompts=list(prompts), context=list(positives), **common)
        neg_inputs = ForwardInput(prompts=list(prompts), context=list(negatives), **common)
        return pos_inputs, neg_inputs

    def _infer(
        self,
        *,
        model,
        rng,
        pos_inputs: ForwardInput,
        neg_inputs: ForwardInput,
        latent_x0s: list[torch.Tensor] | None = None,
        rngs: Sequence[Any] | None = None,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """Causal chunked AR rollout shared by training and validation, with
        return_trajectory=True hardwired and i2v img caches pruned. Returns
        (latents, trajectory_xt, trajectory_pred)."""
        # RAVEN: causal_wan2_1_t2v.py:499-527
        device = get_device()
        # Batched validation passes one rng per prompt; the training path
        # keeps the engine's single rng. Samplers without an rng parameter
        # (e.g. deterministic DDIM) ignore it either way.
        step_rng = rngs if rngs is not None else rng
        bsz = int(pos_inputs.batch_size)  # RAVEN keeps the tensor; int() is exact via __index__
        noises = pos_inputs.noises
        chunk_sizes = pos_inputs.chunk_sizes
        independent_first_chunks = pos_inputs.independent_first_chunks
        sinks, window_sizes = pos_inputs.sinks, pos_inputs.window_sizes
        if isinstance(latent_x0s, torch.Tensor):
            latent_x0s = [latent_x0s[i] for i in range(latent_x0s.size(0))]
        if latent_x0s is not None:
            assert len(latent_x0s) == bsz, f"latent_x0s length {len(latent_x0s)} does not match batch size {bsz}"

        seqlens_per_frame = [noise.size(2) * noise.size(3) // (model.patch_size[1] * model.patch_size[2]) for noise in noises]
        seqlens_per_chunk = [seqlen_per_frame * chunk_size for seqlen_per_frame, chunk_size in zip(seqlens_per_frame, chunk_sizes)]
        independent_first_chunks = [independent_first_chunks[i] if independent_first_chunks[i] is not None else chunk_sizes[i] for i in range(bsz)]
        num_chunks = [(noises[i].size(1) - independent_first_chunks[i]) // chunk_sizes[i] + 1 for i in range(bsz)]
        max_num_chunks = torch.tensor(max(num_chunks), dtype=torch.int32, device=device)
        all_reduce_max(max_num_chunks)  # RAVEN: comm.all_reduce(..., MAX)
        max_num_chunks = int(max_num_chunks)

        # prepare first chunk
        # RAVEN: causal_wan2_1_t2v.py:529-556
        noisy_latents = [noise[:, :first_chunk_shift, :, :] for noise, first_chunk_shift in zip(noises, independent_first_chunks)]
        frame_shifts = [[0] * bsz]

        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids = list()
        latent_indexes, latent_seqlens = list(), list()
        noisy_latent_relative_indexes, noisy_latent_seqlens = list(), list()
        for i, (seqlen_per_frame, first_chunk_shift) in enumerate(zip(seqlens_per_frame, independent_first_chunks)):
            latent_seqlen = seqlen_per_frame * first_chunk_shift
            sample_len = latent_seqlen

            sample_lens.append(sample_len)
            position_ids.extend([0] * latent_seqlen)
            latent_indexes.extend(list(range(curr, curr + latent_seqlen)))
            latent_seqlens.append(latent_seqlen)
            noisy_latent_relative_indexes.extend(list(range(curr_noisy, curr_noisy + latent_seqlen)))
            noisy_latent_seqlens.append(latent_seqlen)

            curr += latent_seqlen
            curr_noisy += latent_seqlen

        packed_position_ids = torch.tensor(position_ids, dtype=torch.int32, device=device)
        packed_latent_indexes = torch.tensor(latent_indexes, dtype=torch.int32, device=device)
        packed_latent_seqlens = torch.tensor(latent_seqlens, dtype=torch.int32, device=device)
        packed_noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=device)
        packed_noisy_latent_seqlens = torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=device)

        # inference-only kvcache (i2v img caches pruned)
        # RAVEN: causal_wan2_1_t2v.py:558-590
        self_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        cross_attn_cache = NaiveCache(model.num_layers, bsz, sink=[0] * bsz, window_size=[None] * bsz)
        inputs_caches = [(pos_inputs, self_attn_cache, cross_attn_cache)]
        if neg_inputs is not None:  # neg_inputs=None: neg chain gated off upstream (rollout)
            neg_self_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
            neg_cross_attn_cache = NaiveCache(model.num_layers, bsz, sink=[0] * bsz, window_size=[None] * bsz)
            inputs_caches.append((neg_inputs, neg_self_attn_cache, neg_cross_attn_cache))

        for inp, sa, ca in inputs_caches:
            inp.packed_position_ids = packed_position_ids
            inp.packed_latent_indexes = packed_latent_indexes
            inp.packed_latent_seqlens = packed_latent_seqlens
            inp.packed_noisy_latent_relative_indexes = packed_noisy_latent_relative_indexes
            inp.packed_noisy_latent_seqlens = packed_noisy_latent_seqlens
            inp.sample_lens = sample_lens
            inp.frame_shifts = frame_shifts
            inp.past_key_values_self_attn = sa
            inp.past_key_values_cross_attn = ca

        # infer first chunk
        # RAVEN: causal_wan2_1_t2v.py:592-617 (tqdm dropped: disabled anyway when return_trajectory)
        latents = []
        trajectory_xt = [[] for _ in range(max_num_chunks)]
        trajectory_pred = [[] for _ in range(max_num_chunks)]

        for t in self.sampling_timesteps.timesteps:
            if isinstance(model, FSDPModule):
                model.unshard()  # trigger 1st all-gather earlier

            trajectory_xt[0].append(noisy_latents)
            t = t.expand((bsz,))
            s = self.sampling_timesteps.get_next_timesteps(t)

            pos_inputs.timesteps = t  # RAVEN: set_timesteps (base_meta_model.py:274-280)
            pos_inputs.xts = noisy_latents  # RAVEN: set_noisy_latents (base_meta_model.py:290-296)
            if neg_inputs is not None:
                neg_inputs.timesteps = t
                neg_inputs.xts = noisy_latents
            pred = self._pred_cfg(model, pos_inputs, neg_inputs)

            # RAVEN: step_to (base_meta_model.py:351-366)
            noisy_latents = self.sampler.step_to(
                pred=pred, x_t=pos_inputs.xts, t=pos_inputs.timesteps, s=s, rng=step_rng, seqlens=pos_inputs.seqlens
            )
            trajectory_pred[0].append(pred)  # list of [C, T, H, W]

        # cache first chunk
        # RAVEN: causal_wan2_1_t2v.py:619-629
        latents.append(noisy_latents)
        if isinstance(model, FSDPModule):
            model.unshard()
        cache_latents = noisy_latents if latent_x0s is None else [
            latent_x0s[i][:, :independent_first_chunks[i], :, :]
            for i in range(bsz)
        ]
        pos_inputs.xts = cache_latents
        if neg_inputs is not None:
            neg_inputs.xts = cache_latents
            pos_inputs, neg_inputs = self._cache(model, pos_inputs, neg_inputs)
        else:
            pos_inputs = self._cache(model, pos_inputs)

        # common packed inputs for the rest chunks
        # RAVEN: causal_wan2_1_t2v.py:631-669
        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids = list()
        latent_indexes, latent_seqlens = list(), list()
        noisy_latent_relative_indexes, noisy_latent_seqlens = list(), list()
        for i, latent_seqlen in enumerate(seqlens_per_chunk):
            sample_lens.append(latent_seqlen)
            position_ids.extend([0] * latent_seqlen)
            latent_indexes.extend(list(range(curr, curr + latent_seqlen)))
            latent_seqlens.append(latent_seqlen)
            noisy_latent_relative_indexes.extend(list(range(curr_noisy, curr_noisy + latent_seqlen)))
            noisy_latent_seqlens.append(latent_seqlen)

            curr += latent_seqlen
            curr_noisy += latent_seqlen

        packed_position_ids = torch.tensor(position_ids, dtype=torch.int32, device=device)
        packed_latent_indexes = torch.tensor(latent_indexes, dtype=torch.int32, device=device)
        packed_latent_seqlens = torch.tensor(latent_seqlens, dtype=torch.int32, device=device)
        packed_noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=device)
        packed_noisy_latent_seqlens = torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=device)

        for inp in (pos_inputs, neg_inputs):
            if inp is None:
                continue
            inp.packed_position_ids = packed_position_ids
            inp.packed_latent_indexes = packed_latent_indexes
            inp.packed_latent_seqlens = packed_latent_seqlens
            inp.packed_noisy_latent_relative_indexes = packed_noisy_latent_relative_indexes
            inp.packed_noisy_latent_seqlens = packed_noisy_latent_seqlens
            inp.sample_lens = sample_lens

        # RAVEN: causal_wan2_1_t2v.py:671-720
        for i in range(max_num_chunks - 1):
            # prepare rest chunk
            noisy_latents = [
                noise[:, min(i * chunk_size + first_chunk_shift, noise.size(1) - chunk_size) :
                      min(i * chunk_size + first_chunk_shift, noise.size(1) - chunk_size) + chunk_size, :, :]
                for noise, chunk_size, first_chunk_shift in zip(noises, chunk_sizes, independent_first_chunks)
            ]

            frame_shifts = [[min(i * chunk_size + first_chunk_shift, noise.size(1) - chunk_size)] for noise, chunk_size, first_chunk_shift in zip(noises, chunk_sizes, independent_first_chunks)]
            pos_inputs.frame_shifts = frame_shifts
            if neg_inputs is not None:
                neg_inputs.frame_shifts = frame_shifts

            # infer rest chunk
            for t in self.sampling_timesteps.timesteps:
                if isinstance(model, FSDPModule):
                    model.unshard()

                trajectory_xt[i + 1].append(noisy_latents)
                t = t.expand((bsz,))
                s = self.sampling_timesteps.get_next_timesteps(t)

                pos_inputs.timesteps = t
                pos_inputs.xts = noisy_latents
                if neg_inputs is not None:
                    neg_inputs.timesteps = t
                    neg_inputs.xts = noisy_latents
                pred = self._pred_cfg(model, pos_inputs, neg_inputs)

                noisy_latents = self.sampler.step_to(
                    pred=pred, x_t=pos_inputs.xts, t=pos_inputs.timesteps, s=s, rng=step_rng, seqlens=pos_inputs.seqlens
                )
                trajectory_pred[i + 1].append(pred)  # list of [C, T, H, W]

            # cache rest chunk
            latents.append(noisy_latents)
            if i != max_num_chunks - 2:  # no need to cache the last chunk
                if isinstance(model, FSDPModule):
                    model.unshard()
                if latent_x0s is None:
                    cache_latents = noisy_latents
                else:
                    cache_latents = [
                        latent_x0s[j][:, min(i * chunk_sizes[j] + independent_first_chunks[j], latent_x0s[j].size(1) - chunk_sizes[j]) :
                                           min(i * chunk_sizes[j] + independent_first_chunks[j], latent_x0s[j].size(1) - chunk_sizes[j]) + chunk_sizes[j], :, :]
                        for j in range(bsz)
                    ]
                pos_inputs.xts = cache_latents
                if neg_inputs is not None:
                    neg_inputs.xts = cache_latents
                    pos_inputs, neg_inputs = self._cache(model, pos_inputs, neg_inputs)
                else:
                    pos_inputs = self._cache(model, pos_inputs)

        # aggregate latents
        # RAVEN: causal_wan2_1_t2v.py:722-734
        latents = [torch.cat([latents[i][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]

        trajectory_xt = [
            [torch.cat([trajectory_xt[i][k][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]
            for k in range(len(trajectory_xt[0]))  # i-th chunk, j-th sample, k-th timestep
        ]
        trajectory_pred = [
            [torch.cat([trajectory_pred[i][k][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]
            for k in range(len(trajectory_pred[0]))  # i-th chunk, j-th sample, k-th timestep
        ]
        return latents, trajectory_xt, trajectory_pred

    def _loss_fn(
        self,
        inputs: ForwardInput,
        pred: list[torch.Tensor],
        target: list[torch.Tensor] | None = None,
        key_prefix: str = "losses",
        score_timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """World-normalized per-token sum-MSE (target given) or precomputed
        per-sample losses (target None), with optional chunk-wise weighting.

        chunk_wise_balancing and first_chunk_aux_loss branches are NOT ported
        (config removed for this project)."""
        device = get_device()

        # base loss — RAVEN: wan2_1_t2v.py:129-165 (loss_fn)
        seqlens = torch.tensor([math.prod(pred[i].shape) for i in range(len(pred))], device=device)
        total_seqlens = seqlens.sum()
        handle = all_reduce_sum(total_seqlens, async_op=True)

        if target is not None:
            losses = [F.mse_loss(pred[i].float(), target[i].float().detach(), reduction="none")
                      for i in range(len(pred))]
            losses = torch.stack([loss.sum() for loss in losses])
        else:
            losses = torch.stack([loss.sum() for loss in pred])

        # binned metrics — RAVEN: wan2_1_t2v.py:147-160. Jarvis' AverageMeter has no
        # count-weighted update; per-observation averaging approximates it (metrics only).
        binned_loss_dict, _ = _bin_losses(
            losses=losses.detach(), seqlens=seqlens, values=inputs.timesteps,
            bin_size=100, min_value=0, max_value=1000,
            key_prefix=f"{key_prefix}_in_timesteps",
        )
        get_running_average_meter().update(binned_loss_dict)

        # inputs.timesteps is the sampling grid; score_timesteps is the separate draw
        # the DMD real/fake scores are evaluated at, so it needs its own bins.
        if score_timesteps is not None:
            binned_loss_dict, _ = _bin_losses(
                losses=losses.detach(), seqlens=seqlens, values=score_timesteps,
                bin_size=100, min_value=0, max_value=1000,
                key_prefix=f"{key_prefix}_in_score_timesteps",
            )
            get_running_average_meter().update(binned_loss_dict)

        if inputs.guidance is not None:
            binned_loss_dict, _ = _bin_losses(
                losses=losses.detach(), seqlens=seqlens, values=inputs.guidance,
                bin_size=1.0, min_value=1.0, max_value=10.0,
                key_prefix=f"{key_prefix}_in_guidance",
            )
            get_running_average_meter().update(binned_loss_dict)

        if handle is not None:
            handle.wait()
        loss = (losses / total_seqlens * get_world_size()).sum()

        # chunk-wise weighting — RAVEN: causal_wan2_1_t2v.py:914-1034
        # (chunk_wise_balancing branch dropped)
        apply_chunk_wise_weighting = (
            self.chunk_wise_weighting is not None and key_prefix in self.chunk_wise_weighting
        )
        if self.chunk_wise_weighting is not None and key_prefix not in self.chunk_wise_weighting:
            if key_prefix not in self._warned_weighting:  # RAVEN: log_n_times
                self._warned_weighting.add(key_prefix)
                logger.warning(f"Chunk-wise weighting config provided but no weighting specified for {key_prefix}.")

        if apply_chunk_wise_weighting:
            weighting_cfg = self.chunk_wise_weighting[key_prefix]
            weighted_losses = []
            weighted_seqlens = []

            # RAVEN: causal_wan2_1_t2v.py:933-1025
            for i, pred_i in enumerate(pred):
                first_chunk_size = inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] is not None else inputs.chunk_sizes[i]
                chunk_sizes = [first_chunk_size]
                t_rest = pred_i.size(1) - first_chunk_size
                assert t_rest % inputs.chunk_sizes[i] == 0, f"t_rest {t_rest} not divisible by chunk size {inputs.chunk_sizes[i]}"
                chunk_sizes.extend([inputs.chunk_sizes[i]] * (t_rest // inputs.chunk_sizes[i]))

                chunk_preds = []
                chunk_targets = []
                start_idx = 0
                for chunk_size in chunk_sizes:
                    end_idx = start_idx + chunk_size
                    chunk_preds.append(pred_i[:, start_idx:end_idx, ...])
                    if target is not None:
                        chunk_targets.append(target[i][:, start_idx:end_idx, ...])
                    start_idx = end_idx

                chunk_seqlens = torch.tensor([math.prod(chunk_pred.shape) for chunk_pred in chunk_preds], device=device)
                weights = self._chunk_weights(chunk_seqlens, weighting_cfg)

                if target is not None:
                    chunk_losses = torch.stack([
                        F.mse_loss(chunk_pred.float(), chunk_target.float().detach(), reduction="none").sum()
                        for chunk_pred, chunk_target in zip(chunk_preds, chunk_targets)
                    ])
                else:
                    chunk_losses = torch.stack([chunk_pred.sum() for chunk_pred in chunk_preds])

                weighted_losses.append((weights * chunk_losses).sum())
                weighted_seqlens.append((weights * chunk_seqlens).sum())

            # RAVEN: causal_wan2_1_t2v.py:1027-1034
            weighted_losses = torch.stack(weighted_losses)
            weighted_seqlens = torch.stack(weighted_seqlens)
            total_weighted_seqlens = weighted_seqlens.sum()
            handle = all_reduce_sum(total_weighted_seqlens, async_op=True)

            if handle is not None:
                handle.wait()
            loss = (weighted_losses / total_weighted_seqlens * get_world_size()).sum()

        return loss

    @staticmethod
    def _validation_spec(config: Any, vae: Any, backbone: Any) -> _ValidationSpec:
        validation = config.validation
        data = config.data.args
        num_frames = validation.num_frames
        height = validation.height
        width = validation.width
        stride_t, stride_h, stride_w = vae.stride
        patch_t, patch_h, patch_w = backbone.patch_size
        channels = vae.z_dim

        assert (num_frames - 1) % stride_t == 0
        assert height % stride_h == 0 and width % stride_w == 0
        latent_t = (num_frames - 1) // stride_t + 1
        latent_h, latent_w = height // stride_h, width // stride_w
        assert latent_h % patch_h == 0 and latent_w % patch_w == 0
        assert patch_t == 1
        assert channels == backbone.in_dim == backbone.out_dim

        chunk_size = data.chunk_size
        first_chunk = data.independent_first_chunk
        sink = data.sink
        window_size = data.window_size
        assert data.separated_first_frame is True
        assert 0 < first_chunk <= latent_t
        if chunk_size == 0:
            # The unchunked (GRPO) validation path has no cache windows
            # either; the chunked (DMD) path carries its own.
            assert sink == 0 and window_size is None
        else:
            assert (latent_t - first_chunk) % chunk_size == 0

        seqlen = latent_t * (latent_h // patch_h) * (latent_w // patch_w)
        return _ValidationSpec(
            latent_shape=(channels, latent_t, latent_h, latent_w),
            seqlen=seqlen,
            chunk_size=chunk_size,
            first_chunk=first_chunk,
            sink=sink,
            window_size=window_size,
            guidance_scale=float(validation.guidance_scale),
            fps=int(validation.get("fps", 16)),
            nrow=int(validation.get("nrow", 8)),
            crf=int(validation.get("crf", 18)),
        )

    def _rollout_validation_latents(
        self,
        *,
        backbone: Any,
        sampling_seed: int,
        pos_inputs: ForwardInput,
        neg_inputs: ForwardInput,
        rngs: Sequence[Any] | None = None,
    ) -> list[torch.Tensor]:
        final_x0s, trajectory_xt, trajectory_pred = self._infer(
            model=backbone,
            rng=RandomState(sampling_seed),
            rngs=rngs,
            pos_inputs=self._clone_inputs(pos_inputs),
            neg_inputs=self._clone_inputs(neg_inputs),
            latent_x0s=None,
        )
        del trajectory_xt, trajectory_pred
        return final_x0s

    @staticmethod
    def _decode_validation_latents(vae: Any, final_x0s: list[torch.Tensor]) -> torch.Tensor:
        latents = torch.stack(final_x0s).to(device=get_device(), dtype=vae.param_dtype)
        decoded = vae.decode(latents)
        assert decoded.ndim == 5 and decoded.shape[0] == 1 and decoded.shape[1] == 3
        return decoded.float().clamp(-1, 1).cpu()

    def _encode_prompts(self, *, models: dict[str, Any], texts: list[str]) -> list[torch.Tensor]:
        """Encode ``texts`` with T5, trimmed per-sample to the true token length."""
        # RAVEN: project/meta_models/wan2_1_t2v.py:67-84 (encode_prompts, texts branch;
        # token_ids branch pruned — real-prompt mode only)
        tokenizer = models["tokenizer"]
        text_encoder = models["text_encoder"]
        device = get_device()
        ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = text_encoder(ids, mask)
        # trim padding: downstream consumes variable-length context lists
        return [u[:v] for u, v in zip(context, seq_lens)]

    @staticmethod
    def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        """Move batch tensors (and lists of tensors) to device, non-blocking."""
        # RAVEN: causal_wan2_1_t2v.py:331 (to_device)
        def move(v):
            if isinstance(v, torch.Tensor):
                return v.to(device, non_blocking=True)
            if isinstance(v, list):
                return [move(x) for x in v]
            return v

        return {k: move(v) for k, v in batch.items()}

    @staticmethod
    def _clone_inputs(inputs: ForwardInput) -> ForwardInput:
        """Tensor-level deep copy, mirroring RAVEN's ``deepcopy_with_tensor``:
        tensors are cloned, lists are shallow-copied with tensors cloned, other
        values are shared (caches are None at this point)."""
        def copy(v):
            if isinstance(v, torch.Tensor):
                return v.detach().clone()
            if isinstance(v, list):
                return [copy(x) for x in v]
            return v

        return ForwardInput(**{f.name: copy(getattr(inputs, f.name)) for f in fields(inputs)})

    def _pred(self, model, inputs: ForwardInput, neg_inputs: ForwardInput | None = None) -> list[torch.Tensor]:
        """Single forward pass through CausalWanModel (packed causal path)."""
        # RAVEN: causal_wan2_1_t2v.py:773-884 (pred)
        # check if inference
        use_flex = inputs.past_key_values_self_attn is None

        # check if we need an explicit flex-attention BlockMask
        # RAVEN: causal_wan2_1_t2v.py:785-806
        from utils.flex_attn import FLEX_FLASH_ATTN_AVAILABLE
        if not FLEX_FLASH_ATTN_AVAILABLE and use_flex:  # block mask needed
            attention_mask = self._block_mask(model, inputs)
        else:
            attention_mask = None

        if use_flex:  # training with xts/timestep/guidance in sample-level, not chunk-level
            # split xts/timestep/guidance: teacher-forcing interleave of noisy and clean chunks
            # RAVEN: causal_wan2_1_t2v.py:808-848
            xs, ts, gs = list(), list(), list()
            curr, concat_indexes = 0, list()

            for i, xt in enumerate(inputs.xts):
                first_chunk_shift = inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] else inputs.chunk_sizes[i]

                cond_i = inputs.latents[i]
                concat_index = list()

                # first noisy chunk
                xs.append(xt[:, :first_chunk_shift, ...])
                ts.append(inputs.timesteps[i:i + 1])
                concat_index.append(curr)
                curr += 1

                # first clean chunk
                cond = cond_i[:, :first_chunk_shift, ...]
                xs.append(cond)
                curr += 1

                t_rest = xt.size(1) - first_chunk_shift
                assert t_rest % inputs.chunk_sizes[i] == 0, f"t_rest {t_rest} not divisible by chunk size {inputs.chunk_sizes[i]}"
                num_rest_chunks = t_rest // inputs.chunk_sizes[i]

                for j in range(num_rest_chunks):
                    start_idx = j * inputs.chunk_sizes[i] + first_chunk_shift
                    end_idx = start_idx + inputs.chunk_sizes[i]
                    xs.append(xt[:, start_idx:end_idx, ...])  # noisy chunk
                    concat_index.append(curr)
                    curr += 1
                    if j != num_rest_chunks - 1:
                        cond = cond_i[:, start_idx:end_idx, ...]
                        xs.append(cond)  # clean chunk
                        curr += 1

                ts.extend([inputs.timesteps[i:i + 1]] * num_rest_chunks)
                if inputs.guidance is not None:
                    gs.extend([inputs.guidance[i:i + 1]] * (1 + num_rest_chunks))
                concat_indexes.append(concat_index)

            ts = torch.cat(ts, dim=0)
            if len(gs) > 0:
                gs = torch.cat(gs, dim=0)
            else:
                gs = None
        else:
            xs = inputs.xts
            ts = inputs.timesteps
            gs = inputs.guidance
            concat_indexes = None

        # RAVEN: causal_wan2_1_t2v.py:854-879 (i2v img cache kwargs pruned — model defaults)
        out = model(
            x           = [x.to(dtype=model.param_dtype) for x in xs],
            t           = ts,
            context     = [c.to(dtype=model.param_dtype) for c in inputs.context],
            guidance    = gs,
            # common packed
            packed_position_ids                     = inputs.packed_position_ids,
            packed_latent_indexes                   = inputs.packed_latent_indexes,
            packed_latent_seqlens                   = inputs.packed_latent_seqlens,
            packed_noisy_latent_relative_indexes    = inputs.packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens             = inputs.packed_noisy_latent_seqlens,
            sample_lens                             = inputs.sample_lens,
            frame_shifts                            = list(itertools.chain(*inputs.frame_shifts)),
            # training only
            attention_mask  = attention_mask,
            q_ranges        = inputs.q_ranges,
            k_ranges        = inputs.k_ranges,
            attn_type_map   = inputs.attn_type_map,
            attn_workloads  = inputs.attn_workloads,
            # inference only
            past_key_values_self_attn               = inputs.past_key_values_self_attn,
            update_past_key_values_self_attn        = inputs.update_past_key_values_self_attn,
            past_key_values_cross_attn              = inputs.past_key_values_cross_attn,
            update_past_key_values_cross_attn       = inputs.update_past_key_values_cross_attn,
        )

        # RAVEN: causal_wan2_1_t2v.py:881-884
        return [
            torch.cat([out[concat_index[i]] for i in range(len(concat_index))], dim=1)  # [C, T, H, W]
            for concat_index in concat_indexes
        ] if concat_indexes is not None else out

    def _pred_cfg(self, model, pos_inputs: ForwardInput, neg_inputs: ForwardInput) -> list[torch.Tensor]:
        """Classifier-free guidance: neg + g * (pos - neg), short-circuit if g <= 1."""
        # RAVEN: causal_wan2_1_t2v.py:886-903 (pred_cfg)
        if neg_inputs is None or pos_inputs.guidance is None or not (pos_inputs.guidance > 1.0).any():
            return self._pred(model, pos_inputs, neg_inputs)

        if model.guidance_embeds is not None:
            return self._pred(model, pos_inputs, neg_inputs)

        bsz = int(pos_inputs.batch_size)
        pred_pos = self._pred(model, pos_inputs)
        pred_neg = self._pred(model, neg_inputs)

        guidance = pos_inputs.guidance
        pred_cfg = [pred_neg[i] + guidance[i] * (pred_pos[i] - pred_neg[i]) for i in range(bsz)]
        return pred_cfg

    def _cache(
        self,
        model,
        inputs: ForwardInput,
        neg_inputs: ForwardInput | None = None,
    ) -> ForwardInput | tuple[ForwardInput, ForwardInput]:
        """Run a cache-filling forward on clean chunk latents (no noisy tokens)."""
        # RAVEN: causal_wan2_1_t2v.py:739-771 (cache; i2v img cache flag pruned)
        if (
            neg_inputs is not None
            and inputs.guidance is not None
            and (inputs.guidance > 1.0).any()
            # Tri-state: only legacy external-CFG models (None) consume a negative
            # KV cache. False/True students are single-forward in _pred_cfg, so a
            # neg cache would be built per chunk and never read (pure waste).
            and model.guidance_embeds is None
        ):
            pos_inputs = self._cache(model, inputs)
            neg_inputs = self._cache(model, neg_inputs)
            return pos_inputs, neg_inputs

        # if not cfg_pred, no need to update anything inside neg_inputs
        ori_packed_noisy_latent_relative_indexes = inputs.packed_noisy_latent_relative_indexes
        inputs.packed_noisy_latent_relative_indexes = torch.tensor([], dtype=torch.int32, device=get_device())
        inputs.update_past_key_values_self_attn = True
        inputs.update_past_key_values_cross_attn = True
        _ = self._pred(model, inputs, neg_inputs)
        inputs.packed_noisy_latent_relative_indexes = ori_packed_noisy_latent_relative_indexes
        inputs.update_past_key_values_self_attn = False
        inputs.update_past_key_values_cross_attn = False
        # do not update kvcache for cross attn anymore in the following caching process
        inputs.context = []
        if neg_inputs is not None:
            return inputs, neg_inputs
        else:
            return inputs

    def _pred_flat(self, model, inputs: ForwardInput) -> list[torch.Tensor]:
        """Plain batched forward through a NON-causal WanModel (fake/teacher)."""
        # RAVEN: wan2_1_t2v.py:244-256 (pred)
        return model(
            x        = [x.to(dtype=model.param_dtype) for x in inputs.xts],
            t        = inputs.timesteps,
            context  = [c.to(dtype=model.param_dtype) for c in inputs.context],
            seq_len  = inputs.max_seq_len,
            guidance = inputs.guidance,
        )

    def _pred_cfg_flat(self, model, pos_inputs: ForwardInput, neg_inputs: ForwardInput) -> list[torch.Tensor]:
        """Merged-batch single-forward CFG for a NON-causal WanModel (teacher)."""
        # RAVEN: wan2_1_t2v.py:111-127 (pred_cfg)
        if (
            neg_inputs is None
            or pos_inputs.guidance is None
            or not (pos_inputs.guidance > 1.0).any()
            or getattr(model, "guidance_embeds", None) is not None
        ):
            return self._pred_flat(model, pos_inputs)

        bsz = int(pos_inputs.batch_size)
        # merge_inputs (RAVEN: wan2_1_t2v.py:98-103 + base) inlined for the fields
        # the flat forward consumes: lists concatenated, tensors cat, max_seq_len max.
        merged = ForwardInput(
            xts=list(pos_inputs.xts) + list(neg_inputs.xts),
            timesteps=torch.cat([pos_inputs.timesteps, neg_inputs.timesteps], dim=0),
            context=list(pos_inputs.context) + list(neg_inputs.context),
            max_seq_len=max(pos_inputs.max_seq_len, neg_inputs.max_seq_len),
            guidance=torch.cat([pos_inputs.guidance, neg_inputs.guidance], dim=0),
        )
        pred_all = self._pred_flat(model, merged)
        pred_pos, pred_neg = pred_all[:bsz], pred_all[bsz:]

        guidance = pos_inputs.guidance
        pred_cfg = [pred_neg[i] + guidance[i] * (pred_pos[i] - pred_neg[i]) for i in range(bsz)]
        return pred_cfg

    def _chunk_weights(self, chunk_seqlens: torch.Tensor, weighting_cfg: float) -> torch.Tensor:
        """Per-chunk loss weights from participation, normalized so the weighted
        token count matches the raw token count."""
        # RAVEN: causal_wan2_1_t2v.py:951-1014 — numeric weighting_cfg branches only
        # (string ln_/mode_ strategies are dead in this project's config).
        assert not isinstance(weighting_cfg, str), "only numeric chunk-wise weighting is implemented"
        participation = torch.flip(torch.cumsum(torch.flip(chunk_seqlens, dims=[0]), dim=0), dims=[0]).float()
        participation = participation / participation[0].clamp_min(1.0)

        if weighting_cfg > 0:
            raw_weights = weighting_cfg * participation / (1 + (weighting_cfg - 1) * participation)
        elif weighting_cfg < 0:
            # reverse participation: later chunks (fewer participants) weigh more
            reverse_participation = participation[-1] / participation.clamp_min(1e-6)
            shift = -weighting_cfg
            raw_weights = shift * reverse_participation / (1 + (shift - 1) * reverse_participation)
        else:
            raw_weights = torch.ones_like(participation)

        # RAVEN: causal_wan2_1_t2v.py:1010-1014
        raw_weighted_seqlens = (raw_weights * chunk_seqlens).sum()
        if raw_weighted_seqlens <= 0:
            raise ValueError("Chunk-wise weighting produced non-positive total weight")
        norm = chunk_seqlens.sum() / raw_weighted_seqlens
        return raw_weights * norm

    def _block_mask(self, model, inputs: ForwardInput):
        """Build the flex-attention BlockMask for the packed training forward.

        Mutates ``inputs`` sample_lens/split_lens/attn_modes with padding entries
        (fresh lists, mirroring RAVEN's deepcopy-then-extend) so the padded
        lengths are also what ``_pred`` hands to the model.

        The only thing the length is aligned to is the sequence-parallel world
        size, so the mask built here is already the length ``CausalWanModelSP``
        shards and it has nothing left to rebuild. Building the mask is the meta
        model's job on every line; a model that patches it afterwards silently
        misses the fusion knob that ``create_sparse_mask`` owns.

        It used to round up to FLEX_ATTN_BLOCK_SIZE as well. That was
        unnecessary: BLOCK_SIZE is the BlockMask's internal tiling and does not
        constrain the sequence -- ``create_mask`` evaluates the predicate on
        exactly (Q_LEN, KV_LEN) and pads the bool itself, and although the Triton
        templates do call ``mask_mod`` past the end of a boundary block, they
        take the index modulo the real length first and mask the result out
        after. ``tests/test_flex_mask_padding_necessity.py`` measures that rather
        than trusting it, including a negative control, so if a torch upgrade
        breaks the guarantee that file fails.

        Without sequence parallelism nothing is padded at all now."""
        # RAVEN: causal_wan2_1_t2v.py:787-804
        alignment = 1
        if is_unified_parallel_initialized():
            alignment = get_unified_parallel_world_size()
        seqlen = sum(inputs.sample_lens)
        seqlen_pad = (seqlen + alignment - 1) // alignment * alignment
        pad_len = seqlen_pad - seqlen
        if pad_len > 0:
            inputs.sample_lens = list(inputs.sample_lens) + [pad_len]
            inputs.split_lens = list(inputs.split_lens) + [[pad_len]]
            inputs.attn_modes = list(inputs.attn_modes) + [['causal']]
        split_lens = list(itertools.chain(*inputs.split_lens))
        attn_modes = list(itertools.chain(*inputs.attn_modes))
        seqlen = sum(inputs.sample_lens)
        assert seqlen % alignment == 0, f"seqlen {seqlen} not divisible by alignment {alignment}"
        return create_sparse_mask(inputs.sample_lens, split_lens, attn_modes, get_device(),
                                  sink=inputs.sinks, window_size=inputs.window_sizes,
                                  block_size=FLEX_ATTN_BLOCK_SIZE)
