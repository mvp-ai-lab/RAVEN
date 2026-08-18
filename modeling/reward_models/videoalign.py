"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import math
import types
from collections.abc import Mapping
from typing import Callable, List, Literal, Optional, Union

import torch
import torch.nn as nn
from decord import VideoReader
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    FlashAttentionKwargs,
    Qwen2VLAttention,
    Unpack,
    VisionAttention,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

from . import BaseRewardModel
# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU instrumentation import >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/videoalign.py:33); MFU instrumentation
# (project.utils.mfu) is not ported to this repo, so the import is removed, the CustomFlops mixin
# below is dropped from the class bases, and the _run_mfu_smoke_test helper is removed.
#         from project.utils.mfu import CustomFlops, enable_flops_accumulate, register_flops_hook
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<

VIDEOSCORE_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the frames of a given video and see the text prompt for generating the video,
then give scores based on its {dimension_name}, i.e., {dimension_description}.
Output a float number from 1.0 to 5.0 for this dimension,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 5.0 means Perfect/Real (the video is like a real video).
The text prompt used for generation is "{text_prompt}".
"""

DIMENSION_DESCRIPTIONS = {
    "VQ": ["visual quality", "the quality of the video in terms of clearness, resolution, brightness, and color"],
    "TA": ["text-to-video alignment", "the alignment between the text prompt and the video content and motion"],
    "MQ": ["motion quality", "the quality of the motion in terms of consistency, smoothness, and completeness"],
    "Overall": [
        "Overall Performance",
        "the overall performance of the video in terms of visual quality, text-to-video alignment, and motion quality",
    ],
}

SIMPLE_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
The text prompt used for generation is "{text_prompt}".
"""

DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""

DETAILED_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Visual Quality, Motion Quality, and Text Alignment.
"""

IMAGE_FACTOR = 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# >>>>>>>>>>>>>> RAVEN adaptation: drop CustomFlops mixin >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/videoalign.py:141); CustomFlops comes
# from the unported project.utils.mfu; the wrapper only needs nn.Module.
#         class FlashAttentionForward(nn.Module, CustomFlops):
class FlashAttentionForward(nn.Module):
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
    def tflops(self, args, kwargs, output) -> float:
        query = args[1] if len(args) > 1 else kwargs["query"]
        key = args[2] if len(args) > 2 else kwargs["key"]

        batch_size, num_heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        return 4 * batch_size * num_heads * (q_len / 1e6) * (kv_len / 1e6) * head_dim

    def forward(self, *args, **kwargs):
        return flash_attention_forward(*args, **kwargs)

def custom_qwen2_vl_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    del use_cache
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.config._attn_implementation in {"flash_attention_2", "flash_attention_3"}:
        attention_interface: Callable = self.flash_attention_forward
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def custom_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    del rotary_pos_emb
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    if self.config._attn_implementation in {"flash_attention_2", "flash_attention_3"}:
        attention_interface: Callable = self.flash_attention_forward
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation in {"flash_attention_2", "flash_attention_3"}:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output

def patch_qwen2vl_attention_for_mfu(module: nn.Module):
    for sub_module in module.modules():
        if isinstance(sub_module, (Qwen2VLAttention, VisionAttention)):
            setattr(sub_module, "flash_attention_forward", FlashAttentionForward())
        if isinstance(sub_module, Qwen2VLAttention):
            sub_module.forward = types.MethodType(custom_qwen2_vl_attention_forward, sub_module)
        elif isinstance(sub_module, VisionAttention):
            sub_module.forward = types.MethodType(custom_vision_attention_forward, sub_module)

def build_prompt(prompt, dimension, template_type):
    if isinstance(dimension, list) and len(dimension) > 1:
        dimension_name = ", ".join([DIMENSION_DESCRIPTIONS[d][0] for d in dimension])
        dimension_name = f"overall performance({dimension_name})"
        dimension_description = "the overall performance of the video"
    else:
        if isinstance(dimension, list):
            dimension = dimension[0]
        dimension_name = DIMENSION_DESCRIPTIONS[dimension][0]
        dimension_description = DIMENSION_DESCRIPTIONS[dimension][1]

    if template_type == "none":
        return prompt
    if template_type == "simple":
        return SIMPLE_PROMPT.format(
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            text_prompt=prompt,
        )
    if template_type == "video_score":
        return VIDEOSCORE_QUERY_PROMPT.format(
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            text_prompt=prompt,
        )
    if template_type == "detailed_special":
        return DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=prompt)
    if template_type == "detailed":
        return DETAILED_PROMPT.format(text_prompt=prompt)
    raise ValueError(f"Invalid template type: {template_type}")

def round_by_factor(number: int | float, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int | float, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int | float, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = VIDEO_MIN_PIXELS,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def smart_nframes(
    total_frames: int,
    video_fps: float,
    fps: Optional[float] = None,
    num_frames: Optional[int] = None,
    min_frames: int = FPS_MIN_FRAMES,
    max_frames: Optional[int] = None,
) -> int:
    if num_frames is not None:
        nframes = round_by_factor(num_frames, FRAME_FACTOR)
    else:
        if fps is None:
            raise ValueError("fps or num_frames must be provided for VideoAlign video preprocessing")
        target_fps = fps
        min_frames = ceil_by_factor(min_frames, FRAME_FACTOR)
        max_frames = floor_by_factor(
            min(FPS_MAX_FRAMES, total_frames) if max_frames is None else min(max_frames, total_frames),
            FRAME_FACTOR,
        )
        nframes = total_frames / video_fps * target_fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if nframes > total_frames:
        nframes = total_frames
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes

def sample_video_frames(
    video: torch.Tensor,
    source_fps: float,
    fps: Optional[float],
    num_frames: Optional[int],
    sample_type: str,
    multi_pts_fps: Optional[float],
) -> torch.Tensor:
    total_frames = video.shape[0]
    if sample_type == "uniform":
        nframes = smart_nframes(
            total_frames=total_frames,
            video_fps=source_fps,
            fps=fps,
            num_frames=num_frames,
        )
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    elif sample_type == "multi_pts":
        frames_each_pts = 6
        num_pts = 4
        if multi_pts_fps is None:
            raise ValueError("multi_pts_fps must be provided for VideoAlign multi_pts sampling")
        target_fps = multi_pts_fps
        nframes = int(total_frames * target_fps // source_fps)
        frames_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes - frames_each_pts // 2 - 1)
        pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frames_idx[pt - frames_each_pts // 2: pt + frames_each_pts // 2])
    else:
        raise ValueError(f"Unsupported sample_type: {sample_type}")
    return video[idx]

def process_video_tensor(
    video: torch.Tensor,
    fps: Optional[float] = None,
    num_frames: Optional[int] = None,
    sample_type: str = "uniform",
    multi_pts_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
    min_pixels: int = VIDEO_MIN_PIXELS,
    total_pixels: int = VIDEO_TOTAL_PIXELS,
    max_pixels: Optional[int] = None,
) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with shape [C, T, H, W], got {tuple(video.shape)}")
    if video.shape[0] not in (1, 3):
        raise ValueError(f"Expected channel dimension C to be 1 or 3, got {video.shape[0]}")
    if not torch.is_floating_point(video):
        raise TypeError(f"Expected floating-point video tensor in [-1, 1], got dtype {video.dtype}")
    video_min = video.amin().item()
    video_max = video.amax().item()
    if video_min < -1.0 or video_max > 1.0:
        raise ValueError(
            f"Expected video tensor range in [-1, 1], got min={video_min:.4f}, max={video_max:.4f}"
        )

    # Match VideoAlign-main semantics:
    # - convert decoded [-1, 1] frames into 8-bit pixels
    # - sample frames using fps / nframes
    # - resize in uint8 space before converting back to float
    if num_frames is None and source_fps is None:
        raise ValueError("source_fps must be provided when sampling by fps in VideoAlign preprocessing")
    video = video.permute(1, 0, 2, 3)
    video = ((video + 1.0) * 127.5).round().clamp(0.0, 255.0).to(torch.uint8)
    video = sample_video_frames(
        video=video,
        source_fps=float(source_fps) if source_fps is not None else 0.0,
        fps=fps,
        num_frames=num_frames,
        sample_type=sample_type,
        multi_pts_fps=multi_pts_fps,
    )

    _, _, height, width = video.shape
    if max_pixels is None:
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / video.shape[0] * FRAME_FACTOR), int(min_pixels * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=int(max_pixels),
    )
    return transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

class Qwen2VLRewardModel(Qwen2VLForConditionalGeneration):
    def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
        super().__init__(config)
        patch_qwen2vl_attention_for_mfu(self)
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        del labels, rope_deltas
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask |= input_ids == special_token_id
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        return {"logits": pooled_logits}

def _find_target_linear_names(model: nn.Module, num_lora_modules: int = -1, lora_namespan_exclude=None) -> List[str]:
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_namespan_exclude = lora_namespan_exclude or []
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names

class VideoAlignRewardModel(BaseRewardModel):
    def __init__(
        self,
        base_model_name_or_path: str,
        model_revision: str = "main",
        output_dim: int = 1,
        use_special_tokens: bool = True,
        trust_remote_code: bool = False,
        torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = "bfloat16",
        reward_token: Literal["last", "mean", "special"] = "special",
        max_frame_pixels: int = 200704,
        eval_dim: Union[str, List[str]] = ("VQ", "MQ", "TA"),
        prompt_template_type: str = "detailed_special",
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        sample_type: str = "uniform",
        multi_pts_fps: Optional[float] = None,
        use_norm: bool = True,
        vq_mean: Optional[float] = 3.6757,
        vq_std: Optional[float] = 2.2476,
        mq_mean: Optional[float] = 1.1646,
        mq_std: Optional[float] = 1.3811,
        ta_mean: Optional[float] = 2.8105,
        ta_std: Optional[float] = 2.5121,
        lora_enable: bool = True,
        vision_lora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_namespan_exclude: Optional[List[str]] = None,
        lora_modules_to_save: Optional[List[str]] = None,
        lora_task_type: str = "CAUSAL_LM",
        use_rslora: bool = False,
        num_lora_modules: int = -1,
    ):
        super().__init__()
        if isinstance(eval_dim, tuple):
            eval_dim = list(eval_dim)
        if sample_type == "uniform":
            self.validate_frame_sampling(fps, num_frames, "VideoAlign")
        if sample_type == "multi_pts" and multi_pts_fps is None:
            raise ValueError("multi_pts_fps must be provided for VideoAlign multi_pts sampling")

        self.max_frame_pixels = max_frame_pixels
        self.eval_dim = eval_dim
        self.prompt_template_type = prompt_template_type
        self.fps = fps
        self.num_frames = num_frames
        self.sample_type = sample_type
        self.multi_pts_fps = multi_pts_fps
        self.use_norm = use_norm
        self.inference_config = None
        if None not in (vq_mean, vq_std, mq_mean, mq_std, ta_mean, ta_std):
            self.inference_config = {
                "VQ_mean": vq_mean,
                "VQ_std": vq_std,
                "MQ_mean": mq_mean,
                "MQ_std": mq_std,
                "TA_mean": ta_mean,
                "TA_std": ta_std,
            }

        self.model, self.processor = self._create_model_and_processor(
            base_model_name_or_path=base_model_name_or_path,
            model_revision=model_revision,
            output_dim=output_dim,
            use_special_tokens=use_special_tokens,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            reward_token=reward_token,
            lora_enable=lora_enable,
            vision_lora=vision_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_namespan_exclude=lora_namespan_exclude,
            lora_modules_to_save=lora_modules_to_save,
            lora_task_type=lora_task_type,
            use_rslora=use_rslora,
            num_lora_modules=num_lora_modules,
        )

    def _create_model_and_processor(
        self,
        base_model_name_or_path: str,
        model_revision: str,
        output_dim: int,
        use_special_tokens: bool,
        trust_remote_code: bool,
        torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]],
        reward_token: str,
        lora_enable: bool,
        vision_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_namespan_exclude: Optional[List[str]],
        lora_modules_to_save: Optional[List[str]],
        lora_task_type: str,
        use_rslora: bool,
        num_lora_modules: int,
    ):
        processor = AutoProcessor.from_pretrained(
            base_model_name_or_path,
            padding_side="right",
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

        special_token_ids = None
        if use_special_tokens:
            special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        config = AutoConfig.from_pretrained(
            base_model_name_or_path,
            revision=model_revision,
            trust_remote_code=trust_remote_code,
        )
        config.use_cache = False
        config._attn_implementation = "flash_attention_2"
        if torch_dtype not in ["auto", None]:
            config.torch_dtype = getattr(torch, torch_dtype)

        model = Qwen2VLRewardModel(
            config,
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
        )

        if use_special_tokens:
            model.resize_token_embeddings(len(processor.tokenizer))

        if lora_enable:
            excluded = lora_namespan_exclude or []
            if not vision_lora:
                excluded = list(excluded) + ["visual"]
            target_modules = _find_target_linear_names(
                model,
                num_lora_modules=num_lora_modules,
                lora_namespan_exclude=excluded,
            )
            peft_config = LoraConfig(
                target_modules=target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=lora_task_type,
                use_rslora=use_rslora,
                bias="none",
                modules_to_save=lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)

        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        return model, processor

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            device = next(self.parameters()).device
            return data.to(device=device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("Empty inputs")
        return inputs

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        self.validate_video_prompt_batch(video_tensors, prompts, "VideoAlignRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))

        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "file://dummy_path"},
                        {
                            "type": "text",
                            "text": build_prompt(prompt, self.eval_dim, self.prompt_template_type),
                        },
                    ],
                }
            ]
            for prompt in prompts
        ]
        configured_max_pixels = max(int(self.max_frame_pixels), VIDEO_MIN_PIXELS)
        processed_videos = [
            process_video_tensor(
                tensor,
                max_pixels=configured_max_pixels,
                fps=self.fps,
                num_frames=self.num_frames,
                sample_type=self.sample_type,
                multi_pts_fps=self.multi_pts_fps,
                source_fps=sample_source_fps,
            )
            for tensor, sample_source_fps in zip(video_tensors, source_fps)
        ]
        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=processed_videos,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return self._prepare_inputs(batch)

    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: Optional[bool] = None,
    ):
        if use_norm is None:
            use_norm = self.use_norm
        batch = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        rewards = self.model(**batch, return_dict=True)["logits"]

        if use_norm and self.inference_config is not None:
            vq = (rewards[:, 0] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
            mq = (rewards[:, 1] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
            ta = (rewards[:, 2] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
        else:
            vq = rewards[:, 0]
            mq = rewards[:, 1]
            ta = rewards[:, 2]

        overall = vq + mq + ta
        return {"VQ": vq, "MQ": mq, "TA": ta, "Overall": overall}

    def forward(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: Optional[bool] = None,
        **kwargs,
    ):
        del kwargs
        return self.reward_from_frames(
            video_tensors=video_tensors,
            prompts=prompts,
            source_fps=source_fps,
            use_norm=use_norm,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("base_model.model.model"):
                new_key = "model.base_model.model.model.language_model" + key[len("base_model.model.model"):]
            elif key.startswith("base_model.model.visual"):
                new_key = "model.base_model.model.model.visual" + key[len("base_model.model.visual"):]
            else:
                new_key = f"model.{key}"
            remapped_state_dict[new_key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)

def _load_video_tensor(video_path: str) -> torch.Tensor:
    reader = VideoReader(video_path)
    frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    return video / 127.5 - 1.0

def _format_reward_output(reward_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    formatted = {}
    for key, value in reward_dict.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar reward for {key}, got shape {tuple(value.shape)}")
            formatted[key] = float(value.detach().cpu().item())
        else:
            formatted[key] = float(value)
    return formatted

# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU smoke helper and __main__ smoke block >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/videoalign.py:896-1121);
# _run_mfu_smoke_test depends on the unported project.utils.mfu (register_flops_hook /
# enable_flops_accumulate) and the __main__ smoke block depends on unported
# project.distributed.fsdp / project.engines.base_engine / project.utils.comm /
# project.utils.config, so both are removed.
#         def _run_mfu_smoke_test(model: VideoAlignRewardModel, video_tensor: torch.Tensor, prompt: str, source_fps: float):
#             if not next(model.parameters()).is_cuda:
#                 print("Skipping MFU smoke test because the model is not on CUDA.")
#                 return
#
#             if not hasattr(model, "flops_state"):
#                 register_flops_hook(model, "videoalign_smoke")
#
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)
#
#             torch.cuda.synchronize()
#             with torch.no_grad():
#                 with enable_flops_accumulate():
#                     start.record()
#                     _ = model.reward_from_frames([video_tensor], [prompt], source_fps=source_fps, use_norm=False)
#                     end.record()
#             torch.cuda.synchronize()
#
#             iter_time = start.elapsed_time(end) / 1e3
#             accumulator = model.flops_state.flops_accumulator.accumulator
#             items = list(accumulator.items())
#             total_tflops = sum(v.item() if torch.is_tensor(v) else v for _, v in items)
#             if total_tflops > 0:
#                 table_items = [
#                     (key, f"{value.item() if torch.is_tensor(value) else value:.6f}") for key, value in items
#                 ]
#                 table_items.append(("", ""))
#                 table_items.append(("total", f"{total_tflops:.6f}"))
#                 table = tabulate(
#                     table_items,
#                     headers=[model.flops_state.flops_accumulator.name, "TFLOPs"],
#                     tablefmt="heavy_outline",
#                     numalign="left",
#                     stralign="left",
#                 )
#                 print(table)
#             model.flops_state.flops_accumulator.reset()
#             print(f"VideoAlign MFU smoke test iteration time: {iter_time:.6f}s")
#             print(f"VideoAlign accumulated TFLOPs: {total_tflops:.6f}")
#
#
#         if __name__ == "__main__":
#             import torch.distributed as dist
#             from datetime import timedelta
#             from torch.distributed.device_mesh import init_device_mesh
#             from torch.distributed.distributed_c10d import _set_pg_timeout
#             from tqdm import tqdm
#
#             from project.distributed.fsdp import set_device_mesh, setup_fsdp
#             from project.engines.base_engine import FSDPConfig
#             from project.utils import comm
#             from project.utils.config import CfgNode
#
#             parser = argparse.ArgumentParser(description="Distributed VideoAlign folder evaluation.")
#             parser.add_argument("--video-dir", required=True)
#             parser.add_argument("--src-txt", default="assets/vbench_all_dimension.txt")
#             parser.add_argument("--tgt-txt", default="assets/vbench_self_forcing_extended.txt")
#             parser.add_argument("--model-path", default="/root/models/KlingTeam/VideoReward/checkpoint-11352/model.pth")
#             parser.add_argument("--base-model-path", default="/root/models/Qwen/Qwen2-VL-2B-Instruct")
#             parser.add_argument("--out-jsonl", default=None, help="If set, on global rank 0, write per-video scores to this path (.jsonl).")
#             args = parser.parse_args()
#
#             dist_initialized = False
#             try:
#                 world_size = comm.get_world_size()
#                 rank = comm.get_rank()
#                 local_rank = comm.get_local_rank()
#                 local_world_size = comm.get_local_world_size()
#                 if torch.cuda.is_available():
#                     device = torch.device("cuda", local_rank)
#                     torch.cuda.set_device(device)
#                     torch.cuda.empty_cache()
#                 else:
#                     device = torch.device("cpu")
#                 if world_size > 1:
#                     if device.type != "cuda":
#                         raise RuntimeError("Distributed VideoAlign evaluation requires CUDA.")
#                     dist.init_process_group(
#                         backend="nccl",
#                         rank=rank,
#                         world_size=world_size,
#                         timeout=timedelta(minutes=60),
#                     )
#                     dist_initialized = True
#                     hybrid_gpu_num = local_world_size
#                     if world_size % hybrid_gpu_num != 0:
#                         raise ValueError(f"world_size {world_size} must be divisible by local_world_size {hybrid_gpu_num}")
#                     device_mesh = init_device_mesh(
#                         device_type="cuda",
#                         mesh_shape=(world_size // hybrid_gpu_num, hybrid_gpu_num),
#                         mesh_dim_names=("dp", "fsdp"),
#                     )
#                     _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=0))
#                     _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=1))
#                     set_device_mesh(device_mesh)
#
#                 src_prompts = []
#                 with open(args.src_txt, "r") as f:
#                     src_prompts = [line.strip() for line in f]
#                 tgt_prompts = []
#                 with open(args.tgt_txt, "r") as f:
#                     tgt_prompts = [line.strip() for line in f]
#                 if len(src_prompts) != len(tgt_prompts):
#                     raise ValueError(f"src/tgt prompt count mismatch: {len(src_prompts)} vs {len(tgt_prompts)}")
#
#                 video_paths = sorted(
#                     os.path.join(args.video_dir, entry.name)
#                     for entry in os.scandir(args.video_dir)
#                     if entry.is_file() and os.path.splitext(entry.name)[1].lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
#                 )
#                 if not video_paths:
#                     raise FileNotFoundError(f"No videos found under {args.video_dir}")
#
#                 samples = []
#                 errors = []
#                 for video_path in video_paths:
#                     stem = os.path.splitext(os.path.basename(video_path))[0]
#                     matches = [(idx, src_prompt) for idx, src_prompt in enumerate(src_prompts) if src_prompt and stem.startswith(src_prompt)]
#                     if not matches:
#                         errors.append(f"{video_path}: no src prompt matched by startswith")
#                         continue
#                     max_len = max(len(src_prompt) for _, src_prompt in matches)
#                     matches = [(idx, src_prompt) for idx, src_prompt in matches if len(src_prompt) == max_len]
#                     prompt_index, source_prompt = matches[-1]
#                     samples.append(
#                         {
#                             "video_path": video_path,
#                             "prompt_index": prompt_index,
#                             "source_prompt": source_prompt,
#                             "prompt": tgt_prompts[prompt_index],
#                         }
#                     )
#                 if errors:
#                     raise ValueError("Failed to resolve prompts:\n" + "\n".join(errors))
#
#                 model_config = CfgNode.parse_config("configs/models/videoalign.jsonc")
#                 model_config["base_model_name_or_path"] = args.base_model_path
#                 model = VideoAlignRewardModel(**{k: v for k, v in model_config.items() if not k.startswith("_")})
#                 state_dict = torch.load(args.model_path, map_location="cpu", mmap=True)
#                 msg = model.load_state_dict(state_dict, strict=False, assign=True)
#                 if rank == 0:
#                     print(f"Loaded weight with missing keys: {msg.missing_keys}")
#                     print(f"Loaded weight with unexpected keys: {msg.unexpected_keys}")
#
#                 for name, module in model.named_modules():
#                     module.layer_name = f"reward_model_videoalign.{name}" if name else "reward_model_videoalign"
#                 model.eval().requires_grad_(False)
#                 if world_size > 1:
#                     model = setup_fsdp(
#                         model,
#                         FSDPConfig(enabled=True, auto_wrap_policy="qwen2_vl_wrap_policy", weight_dtype="bfloat16"),
#                         "reward_model_videoalign",
#                     )
#                 elif device.type == "cuda":
#                     model.to(device=device, dtype=torch.bfloat16)
#                 else:
#                     model.to(device=device)
#                 model.eval().requires_grad_(False)
#
#                 local_samples = samples[rank::world_size]
#                 max_steps = len(local_samples)
#                 if world_size > 1:
#                     step_tensor = torch.tensor([max_steps], device=device, dtype=torch.int64)
#                     dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
#                     max_steps = int(step_tensor.item())
#
#                 local_results = []
#                 dummy_sample = samples[0]
#                 for step in tqdm(range(max_steps), disable=local_rank != 0, desc=f"VideoAlign scoring node {rank // local_world_size}"):
#                     sample = local_samples[step] if step < len(local_samples) else dummy_sample
#                     video_tensor = _load_video_tensor(sample["video_path"])
#                     source_fps = float(VideoReader(sample["video_path"]).get_avg_fps())
#                     with torch.inference_mode():
#                         rewards = model(
#                             video_tensors=[video_tensor],
#                             prompts=[sample["prompt"]],
#                             source_fps=[source_fps],
#                             use_norm=True,
#                         )
#                     if step >= len(local_samples):
#                         continue
#                     local_results.append(
#                         {
#                             "video_path": sample["video_path"],
#                             "prompt_index": sample["prompt_index"],
#                             "source_prompt": sample["source_prompt"],
#                             "prompt": sample["prompt"],
#                             "scores": _format_reward_output(rewards),
#                         }
#                     )
#
#                 gathered = comm.all_gather_object(local_results)
#
#                 if local_rank == 0:
#                     results = []
#                     for shard in gathered:
#                         results.extend(shard)
#                     results.sort(key=lambda item: item["video_path"])
#                     summary = {
#                         "node_rank": rank // local_world_size,
#                         "video_dir": args.video_dir,
#                         "src_txt": args.src_txt,
#                         "tgt_txt": args.tgt_txt,
#                         "video_count": len(results),
#                     }
#                     summary["mean_scores"] = {
#                         key: sum(item["scores"][key] for item in results) / len(results)
#                         for key in ["VQ", "MQ", "TA", "Overall"]
#                     }
#                     print(
#                         json.dumps(
#                             summary,
#                             ensure_ascii=False,
#                             indent=2,
#                         )
#                     )
#                     if args.out_jsonl is not None and rank == 0:
#                         os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
#                         with open(args.out_jsonl, "w") as f:
#                             for item in results:
#                                 f.write(json.dumps(item, ensure_ascii=False) + "\n")
#                         print(f"wrote {len(results)} results to {args.out_jsonl}")
#             finally:
#                 if dist_initialized and dist.is_initialized():
#                     dist.destroy_process_group()
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
