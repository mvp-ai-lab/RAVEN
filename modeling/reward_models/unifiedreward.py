"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com

Standalone distributed folder evaluation entry, launched from the repo root as:

    torchrun --nnodes=... --nproc_per_node=... \
        -m modeling.reward_models.unifiedreward \
        --video-dir <dir>

(same ``-m`` package path convention as ``dev.yanzuolu.common.launch`` in
dev/yanzuolu/tools/multi_run.sh:73)
"""
import argparse
import glob
import logging
import math
import os
import re
from typing import List, Optional, Tuple, Union

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from . import BaseRewardModel
# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU instrumentation imports >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:7,11,12,15,16-21,25);
# MFU instrumentation (project.utils.mfu) is not ported to this repo, so its import and the
# attention-patching machinery below are removed. `types` / `torch.nn` were only used by that
# machinery, `tabulate` was unused upstream, and the transformers flash-attention /
# Qwen2.5-VL internals were only imported to rebuild the patched forwards. Dropping them makes
# the model take the stock HF `flash_attention_2` path -- numerically identical, no MFU counters.
#         import types
#         import torch.nn as nn
#         from tabulate import tabulate
#         from transformers.integrations.flash_attention import flash_attention_forward
#         from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
#             Qwen2_5_VLAttention,
#             Qwen2_5_VLVisionAttention,
#             apply_multimodal_rotary_pos_emb,
#             apply_rotary_pos_emb_vision,
#         )
#         from project.utils.mfu import CustomFlops, enable_flops_accumulate, register_flops_hook
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MODEL_SEQ_LEN = int(float(os.environ.get("MODEL_SEQ_LEN", 128000)))
logger = logging.getLogger(__name__)


# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU attention patching >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:38-153); the
# FlashAttentionForward wrapper and the two custom Qwen2.5-VL attention forwards exist purely to
# count flops through project.utils.mfu.CustomFlops, which is not ported. Without the patch the
# model runs HF's own flash_attention_2 kernels, so the removal is numerically transparent.
#         class FlashAttentionForward(nn.Module, CustomFlops):
#             def tflops(self, args, kwargs, output) -> float:
#                 del output
#                 query = args[1] if len(args) > 1 else kwargs["query"]
#                 key = args[2] if len(args) > 2 else kwargs["key"]
#                 batch_size, num_heads, q_len, head_dim = query.shape
#                 kv_len = key.shape[2]
#                 return 4 * batch_size * num_heads * (q_len / 1e6) * (kv_len / 1e6) * head_dim
#
#             def forward(self, *args, **kwargs):
#                 return flash_attention_forward(*args, **kwargs)
#
#
#         def _custom_qwen2_5_vl_attention_forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values=None,
#             output_attentions: bool = False,
#             use_cache: bool = False,
#             cache_position: Optional[torch.LongTensor] = None,
#             position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
#             **kwargs,
#         ):
#             del use_cache, output_attentions
#             bsz, q_len, _ = hidden_states.size()
#
#             query_states = self.q_proj(hidden_states)
#             key_states = self.k_proj(hidden_states)
#             value_states = self.v_proj(hidden_states)
#
#             query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#             key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#             value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#
#             cos, sin = position_embeddings
#             query_states, key_states = apply_multimodal_rotary_pos_emb(
#                 query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
#             )
#
#             if past_key_values is not None:
#                 cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#                 key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
#
#             attn_output, attn_weights = self.flash_attention_forward(
#                 self,
#                 query_states,
#                 key_states,
#                 value_states,
#                 attention_mask,
#                 dropout=0.0 if not self.training else self.attention_dropout,
#                 scaling=self.scaling,
#                 sliding_window=self.sliding_window,
#                 position_ids=position_ids,
#                 **kwargs,
#             )
#
#             attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#             attn_output = self.o_proj(attn_output)
#             return attn_output, attn_weights
#
#
#         def _custom_qwen2_5_vl_vision_attention_forward(
#             self,
#             hidden_states: torch.Tensor,
#             cu_seqlens: torch.Tensor,
#             rotary_pos_emb: Optional[torch.Tensor] = None,
#             position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
#             **kwargs,
#         ) -> torch.Tensor:
#             del rotary_pos_emb
#             seq_length = hidden_states.shape[0]
#             query_states, key_states, value_states = (
#                 self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#             )
#             cos, sin = position_embeddings
#             query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
#
#             query_states = query_states.transpose(0, 1).unsqueeze(0)
#             key_states = key_states.transpose(0, 1).unsqueeze(0)
#             value_states = value_states.transpose(0, 1).unsqueeze(0)
#
#             max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
#             attn_output, _ = self.flash_attention_forward(
#                 self,
#                 query_states,
#                 key_states,
#                 value_states,
#                 attention_mask=None,
#                 scaling=self.scaling,
#                 dropout=0.0 if not self.training else self.attention_dropout,
#                 cu_seq_lens_q=cu_seqlens,
#                 cu_seq_lens_k=cu_seqlens,
#                 max_length_q=max_seqlen,
#                 max_length_k=max_seqlen,
#                 is_causal=False,
#                 **kwargs,
#             )
#
#             attn_output = attn_output.reshape(seq_length, -1).contiguous()
#             attn_output = self.proj(attn_output)
#             return attn_output
#
#
#         def patch_qwen2_5_vl_attention_for_mfu(module: nn.Module):
#             for sub_module in module.modules():
#                 if not isinstance(sub_module, (Qwen2_5_VLAttention, Qwen2_5_VLVisionAttention)):
#                     continue
#                 if getattr(sub_module.config, "_attn_implementation", None) not in {"flash_attention_2", "flash_attention_3"}:
#                     continue
#                 setattr(sub_module, "flash_attention_forward", FlashAttentionForward())
#                 if isinstance(sub_module, Qwen2_5_VLAttention):
#                     sub_module.forward = types.MethodType(_custom_qwen2_5_vl_attention_forward, sub_module)
#                 else:
#                     sub_module.forward = types.MethodType(_custom_qwen2_5_vl_vision_attention_forward, sub_module)
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<


UNIFIEDREWARD_POINT_SCORE_PROMPT = """Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the frames of a given video and see the text prompt for generating the video.
Then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, the consistency of objects or humans in video
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a number from [1,2,3,4],
in which '1' means 'Bad', '2' means 'Average', '3' means 'Good',
'4' means 'Real' or 'Perfect' (the video is like a real video)
Finally, based on above 5 dimensions, assign a score from 1 to 10 after 'Final Score:'
Here is an output example:
visual quality: 4
temporal consistency: 4
dynamic degree: 3
text-to-video alignment: 1
factual consistency: 2
Final Score: 6

**Note: In the example above, scores are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given video.**
Your task is provided as follows: Text Prompt: [{text_prompt}]"""


def _build_messages(prompts: List[str], video_paths: Optional[List[str]] = None):
    if video_paths is None:
        video_paths = ["file://dummy_path"] * len(prompts)
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": UNIFIEDREWARD_POINT_SCORE_PROMPT.format(text_prompt=prompt)},
                ],
            }
        ]
        for prompt, video_path in zip(prompts, video_paths)
    ]


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    if min_pixels is None:
        raise ValueError("min_pixels must be provided for UnifiedReward video preprocessing")
    if max_pixels is None:
        raise ValueError("max_pixels must be provided for UnifiedReward video preprocessing")
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = _floor_by_factor(height / scale, factor)
        resized_width = _floor_by_factor(width / scale, factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * scale, factor)
        resized_width = _ceil_by_factor(width * scale, factor)
    return resized_height, resized_width


def _smart_nframes(total_frames: int, video_fps: float, fps: Optional[float], nframes: Optional[int]) -> int:
    if nframes is not None:
        sample_frames = _round_by_factor(nframes, FRAME_FACTOR)
    else:
        if fps is None:
            raise ValueError("fps or nframes must be provided for UnifiedReward video preprocessing")
        target_fps = fps
        min_frames = _ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
        max_frames = _floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
        sample_frames = total_frames / video_fps * target_fps
        sample_frames = min(min(max(sample_frames, min_frames), max_frames), total_frames)
        sample_frames = _floor_by_factor(sample_frames, FRAME_FACTOR)
    if not (FRAME_FACTOR <= sample_frames <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {sample_frames}.")
    return int(sample_frames)


def _process_video_tensor(
    tensor: torch.Tensor,
    source_fps: float,
    fps: Optional[float],
    nframes: Optional[int],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    image_patch_size: int,
):
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor with shape [C, T, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[0] not in (1, 3):
        raise ValueError(f"Expected channel dimension C to be 1 or 3, got {tensor.shape[0]}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"Expected floating-point video tensor in [-1, 1], got dtype {tensor.dtype}")
    video_min = tensor.amin().item()
    video_max = tensor.amax().item()
    if video_min < -1.0 or video_max > 1.0:
        raise ValueError(f"Expected video tensor range in [-1, 1], got min={video_min:.4f}, max={video_max:.4f}")

    video = ((tensor.detach().float() + 1.0) * 127.5).round().clamp(0.0, 255.0).to(torch.uint8)
    video = video.permute(1, 0, 2, 3).contiguous()
    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1)

    total_frames = video.shape[0]
    sample_frames = _smart_nframes(total_frames, source_fps, fps=fps, nframes=nframes)
    idx = torch.linspace(0, total_frames - 1, sample_frames, device=video.device).round().long()
    video = video[idx]
    sample_fps = sample_frames / max(total_frames, 1e-6) * source_fps

    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    sample_frames, _, height, width = video.shape
    min_pixels = video_frame_min_pixels if min_pixels is None else min_pixels
    total_pixels = MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    max_pixels_limit = max(min(video_frame_max_pixels, total_pixels / sample_frames * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = max_pixels_limit if max_pixels is None else min(max_pixels, max_pixels_limit)
    resized_height, resized_width = _smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, sample_fps


def _prepare_batch_from_frames(
    processor,
    video_tensors: List[torch.Tensor],
    prompts: List[str],
    source_fps: Union[float, List[Optional[float]]],
    fps: Optional[float],
    nframes: Optional[int],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    device: Optional[torch.device] = None,
):
    BaseRewardModel.validate_video_prompt_batch(video_tensors, prompts, "UnifiedRewardQwenPointScoreRewardModel")
    source_fps = BaseRewardModel.normalize_source_fps(source_fps, len(video_tensors))

    processed_videos = []
    sampled_fps = []
    image_patch_size = int(processor.video_processor.patch_size)
    for tensor, sample_source_fps in zip(video_tensors, source_fps):
        if sample_source_fps is None:
            raise ValueError("source_fps must be provided for UnifiedReward video preprocessing")
        video, sample_fps = _process_video_tensor(
            tensor,
            source_fps=float(sample_source_fps),
            fps=fps,
            nframes=nframes,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_patch_size=image_patch_size,
        )
        processed_videos.append(video)
        sampled_fps.append(sample_fps)

    videos_kwargs = {"do_sample_frames": False, "fps": sampled_fps}
    batch = processor(
        text=processor.apply_chat_template(_build_messages(prompts), tokenize=False, add_generation_prompt=True),
        images=None,
        videos=processed_videos,
        padding=True,
        return_tensors="pt",
        videos_kwargs=videos_kwargs,
    )
    if device is not None:
        batch = batch.to(device=device)
    return batch


class UnifiedRewardQwenPointScoreRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        torch_dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
        device_map: Optional[Union[str, dict]] = None,
        padding_side: str = "left",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        max_new_tokens: int = 128,
        generation_kwargs: Optional[dict] = None,
        strict_parse: bool = False,
        parse_default_scores: Optional[dict] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "UnifiedReward")
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
        # >>>>>>>>>>>>>> RAVEN adaptation: drop MFU attention patch call >>>>>>>>>>>>>>
        # NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:387); the
        # patch helper is removed above together with project.utils.mfu.
        #         patch_qwen2_5_vl_attention_for_mfu(self.model)
        # <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self.processor.tokenizer.padding_side = padding_side
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.generation_kwargs = {"max_new_tokens": max_new_tokens}
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)
        self.strict_parse = strict_parse
        self.parse_default_scores = {"VQ": 2.5, "TC": 2.5, "DD": 2.5, "TA": 2.5, "FC": 2.5, "Overall": 5.0}
        if parse_default_scores is not None:
            self.parse_default_scores.update(parse_default_scores)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        return _prepare_batch_from_frames(
            self.processor,
            video_tensors,
            prompts,
            source_fps,
            fps=self.fps,
            nframes=self.num_frames,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            device=next(self.parameters()).device,
        )

    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: bool = True,
    ):
        del use_norm
        batch = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        generation_kwargs = dict(self.generation_kwargs)
        if (
            "synced_gpus" not in generation_kwargs
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            generation_kwargs["synced_gpus"] = True
        with torch.no_grad():
            generated_ids = self.model.generate(**batch, **generation_kwargs)

        generated_ids = generated_ids[:, batch.input_ids.shape[1]:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        patterns = {
            "VQ": r"visual\s+quality\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
            "TC": r"temporal\s+consistency\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
            "DD": r"dynamic\s+degree\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
            "TA": r"text\s*[- ]?\s*to\s*[- ]?\s*video\s+alignment\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
            "FC": r"factual\s+consistency\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
            "Overall": r"final\s+score\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
        }
        values = {key: [] for key in patterns}
        for output_text in output_texts:
            missing_keys = []
            for key, pattern in patterns.items():
                matches = re.findall(pattern, output_text, flags=re.IGNORECASE)
                if not matches:
                    if self.strict_parse:
                        raise ValueError(f"UnifiedReward failed to parse {key} from output: {output_text}")
                    missing_keys.append(key)
                    values[key].append(float(self.parse_default_scores[key]))
                else:
                    values[key].append(float(matches[-1]))
            if missing_keys:
                logger.warning(
                    "UnifiedReward failed to parse %s from output, using defaults: %s",
                    missing_keys,
                    output_text[:1000],
                )

        device = batch.input_ids.device
        return {key: torch.tensor(value, device=device, dtype=torch.float32) for key, value in values.items()}

    def forward(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: bool = True,
        **kwargs,
    ):
        del kwargs
        return self.reward_from_frames(
            video_tensors=video_tensors,
            prompts=prompts,
            source_fps=source_fps,
            use_norm=use_norm,
        )


def _load_video_tensor(video_path: str):
    from decord import VideoReader

    reader = VideoReader(video_path)
    frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    return video / 127.5 - 1.0, float(reader.get_avg_fps())


def _check_preprocess_batch(batch):
    for key in ["input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]:
        value = batch[key]
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if value.numel() == 0:
            raise AssertionError(f"{key} is empty")
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")


if __name__ == "__main__":
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from tqdm import tqdm

    # >>>>>>>>>>>>>> RAVEN adaptation: repoint distributed/config helpers >>>>>>>>>>>>>>
    # NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:521-530);
    # project.utils.comm -> dev.yanzuolu.common.distributed.ops (same rank/size/all_gather_object
    # semantics), project.utils.config.CfgNode -> dev.yanzuolu.common.config.CfgNode (yaml
    # `from_file` instead of jsonc `parse_config`), and project.distributed.fsdp /
    # project.engines.base_engine are replaced by FSDP2 `fully_shard` below.
    #         from torch.distributed.device_mesh import init_device_mesh
    #         from torch.distributed.distributed_c10d import _set_pg_timeout
    #
    #         from project.distributed.fsdp import set_device_mesh, setup_fsdp
    #         from project.engines.base_engine import FSDPConfig
    #         from project.utils import comm
    #         from project.utils.config import CfgNode
    # <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
    from common.config import CfgNode
    from common.distributed import ops
    from common.model.placement import _get_hsdp_mesh

    MODEL_CONFIG_PATH = "configs/reward_models/unifiedreward.yaml"
    WRAP_CLASSES = {"Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"}

    parser = argparse.ArgumentParser(description="Distributed UnifiedReward folder evaluation.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--src-txt", default="third_party/self_forcing/assets/vbench_all_dimension.txt")
    parser.add_argument("--tgt-txt", default="third_party/self_forcing/assets/vbench_self_forcing_extended.txt")
    parser.add_argument("--model-path", default="/home/jinchengy/jinchengy/yanzuolu/models/CodeGoat24/UnifiedReward-qwen-32b")
    args = parser.parse_args()

    dist_initialized = False
    try:
        world_size = ops.get_world_size()
        rank = ops.get_rank()
        local_rank = ops.get_local_rank()
        local_world_size = ops.get_local_world_size()
        # Same guard as placement.py:313-315: ops defaults LOCAL_WORLD_SIZE to 1, which
        # would put every rank on cuda:0 with an unsharded copy of the model.
        assert world_size == 1 or "LOCAL_WORLD_SIZE" in os.environ, "LOCAL_WORLD_SIZE must be set when world size is greater than 1"
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
        if world_size > 1:
            if device.type != "cuda":
                raise RuntimeError("Distributed UnifiedReward evaluation requires CUDA.")
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                device_id=device,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=60),
            )
            dist_initialized = True
            hybrid_gpu_num = local_world_size
            if world_size % hybrid_gpu_num != 0:
                raise ValueError(f"world_size {world_size} must be divisible by local_world_size {hybrid_gpu_num}")

        src_prompts = []
        with open(args.src_txt, "r") as f:
            src_prompts = [line.strip() for line in f]
        tgt_prompts = []
        with open(args.tgt_txt, "r") as f:
            tgt_prompts = [line.strip() for line in f]
        if len(src_prompts) != len(tgt_prompts):
            raise ValueError(f"src/tgt prompt count mismatch: {len(src_prompts)} vs {len(tgt_prompts)}")

        video_paths = sorted(
            os.path.join(args.video_dir, entry.name)
            for entry in os.scandir(args.video_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
        )
        if not video_paths:
            raise FileNotFoundError(f"No videos found under {args.video_dir}")

        samples = []
        errors = []
        for video_path in video_paths:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            matches = [(idx, src_prompt) for idx, src_prompt in enumerate(src_prompts) if src_prompt and stem.startswith(src_prompt)]
            if not matches:
                errors.append(f"{video_path}: no src prompt matched by startswith")
                continue
            max_len = max(len(src_prompt) for _, src_prompt in matches)
            matches = [(idx, src_prompt) for idx, src_prompt in matches if len(src_prompt) == max_len]
            prompt_index, source_prompt = matches[-1]
            samples.append(
                {
                    "video_path": video_path,
                    "prompt_index": prompt_index,
                    "source_prompt": source_prompt,
                    "prompt": tgt_prompts[prompt_index],
                }
            )
        if errors:
            raise ValueError("Failed to resolve prompts:\n" + "\n".join(errors))

        model_config = CfgNode.from_file(MODEL_CONFIG_PATH)
        model_config["attn_implementation"] = "flash_attention_2"
        model = UnifiedRewardQwenPointScoreRewardModel.from_pretrained(
            args.model_path,
            **{k: v for k, v in model_config.items() if not k.startswith("_")},
        )
        # >>>>>>>>>>>>>> RAVEN adaptation: drop layer_name tagging >>>>>>>>>>>>>>
        # NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:618-619);
        # layer_name only labels modules for the unported project.utils.mfu flops hook.
        #         for name, module in model.named_modules():
        #             module.layer_name = f"reward_model_unified.{name}" if name else "reward_model_unified"
        # <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
        model.eval().requires_grad_(False)
        if world_size > 1:
            # >>>>>>>>>>>>>> RAVEN adaptation: setup_fsdp -> FSDP2 fully_shard >>>>>>>>>>>>>>
            # NOTE: upstream 原样 (RAVEN: project/models/reward_models/unifiedreward.py:564-571,
            # 622-626); RAVEN's setup_fsdp + "qwen2_5_vl_wrap_policy" + the (dp, fsdp) mesh are
            # replaced by the local FSDP2 convention (dev/yanzuolu/common/model/placement.py:293-380):
            # a (replicate, shard) hybrid mesh only when the world spans nodes, otherwise the
            # default mesh. Model is eval + requires_grad_(False), so no trainable-param cast.
            #         device_mesh = init_device_mesh(
            #             device_type="cuda",
            #             mesh_shape=(world_size // hybrid_gpu_num, hybrid_gpu_num),
            #             mesh_dim_names=("dp", "fsdp"),
            #         )
            #         _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=0))
            #         _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=1))
            #         set_device_mesh(device_mesh)
            #         ...
            #         model = setup_fsdp(
            #             model,
            #             FSDPConfig(enabled=True, auto_wrap_policy="qwen2_5_vl_wrap_policy", weight_dtype="bfloat16"),
            #             "reward_model_unified",
            #         )
            # <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
            fsdp_kwargs = {
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=False,
                ),
                "reshard_after_forward": True,
            }
            if local_world_size < world_size:
                fsdp_kwargs["mesh"] = _get_hsdp_mesh((world_size // local_world_size, local_world_size))
            wrapped = set()
            for module in model.modules():
                class_name = type(module).__name__
                if class_name in WRAP_CLASSES:
                    fully_shard(module, **fsdp_kwargs)
                    wrapped.add(class_name)
            # Same guard as placement.py:350-352: a wrap class matching nothing would
            # silently shard the whole 32B as one unit and OOM on the first all-gather.
            assert wrapped == WRAP_CLASSES, f"{sorted(WRAP_CLASSES - wrapped)} matched no submodule of the model"
            fully_shard(model, **fsdp_kwargs)
        elif device.type == "cuda":
            model.to(device=device, dtype=torch.bfloat16)
        else:
            model.to(device=device)
        model.eval().requires_grad_(False)

        local_samples = samples[rank::world_size]
        max_steps = len(local_samples)
        if world_size > 1:
            step_tensor = torch.tensor([max_steps], device=device, dtype=torch.int64)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
            max_steps = int(step_tensor.item())

        local_results = []
        dummy_sample = samples[0]
        for step in tqdm(range(max_steps), disable=local_rank != 0, desc=f"UnifiedReward scoring node {rank // local_world_size}"):
            sample = local_samples[step] if step < len(local_samples) else dummy_sample
            video_tensor, source_fps = _load_video_tensor(sample["video_path"])
            # inference_mode (upstream) poisons FSDP2's _unsharded_param: the second
            # all-gather reads tensor._version and raises on inference tensors.
            with torch.no_grad():
                rewards = model(
                    video_tensors=[video_tensor],
                    prompts=[sample["prompt"]],
                    source_fps=[source_fps],
                    use_norm=False,
                )
            if step >= len(local_samples):
                continue
            local_results.append(
                {
                    "video_path": sample["video_path"],
                    "prompt_index": sample["prompt_index"],
                    "source_prompt": sample["source_prompt"],
                    "prompt": sample["prompt"],
                    "scores": {
                        key: float(value.detach().cpu().item())
                        for key, value in rewards.items()
                    },
                }
            )

        gathered = ops.all_gather_object(local_results)

        if rank == 0:
            import json
            results = []
            for shard in gathered:
                results.extend(shard)
            results.sort(key=lambda item: item["video_path"])
            summary = {
                "node_rank": rank // local_world_size,
                "video_dir": args.video_dir,
                "src_txt": args.src_txt,
                "tgt_txt": args.tgt_txt,
                "video_count": len(results),
            }
            summary["mean_scores"] = {
                key: sum(item["scores"][key] for item in results) / len(results)
                for key in ["VQ", "TC", "DD", "TA", "FC", "Overall"]
            }
            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        if dist_initialized and dist.is_initialized():
            dist.destroy_process_group()
