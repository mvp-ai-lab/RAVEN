import os
import types
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from . import BaseRewardModel

# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU instrumentation import >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_aesthetic_reward.py:11); MFU
# instrumentation (project.utils.mfu) is not ported to this repo, so the import is removed and
# the CustomFlops mixins below are dropped from the class bases.
#         from project.utils.mfu import CustomFlops
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<


# >>>>>>>>>>>>>> RAVEN adaptation: drop CustomFlops mixin >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_aesthetic_reward.py:14);
# CustomFlops comes from the unported project.utils.mfu; the wrapper only needs nn.Module.
#         class MultiheadAttentionForward(nn.Module, CustomFlops):
class MultiheadAttentionForward(nn.Module):
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
    def tflops(self, args, kwargs, output) -> float:
        del output
        query = args[1] if len(args) > 1 else kwargs["query"]
        key = args[2] if len(args) > 2 else kwargs["key"]
        seq_len, batch_size, dim = query.shape
        kv_len = key.shape[0]
        num_heads = args[0].num_heads if len(args) > 0 else kwargs["attn_module"].num_heads
        head_dim = dim // num_heads
        return 4 * batch_size * num_heads * (seq_len / 1e6) * (kv_len / 1e6) * head_dim

    def forward(self, attn_module, query, key, value, **kwargs):
        return attn_module(query, key, value, **kwargs)


# >>>>>>>>>>>>>> RAVEN adaptation: drop CustomFlops mixin >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_aesthetic_reward.py:29);
# CustomFlops comes from the unported project.utils.mfu; the wrapper only needs nn.Module.
#         class FunctionalMultiheadAttentionForward(nn.Module, CustomFlops):
class FunctionalMultiheadAttentionForward(nn.Module):
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
    def tflops(self, args, kwargs, output) -> float:
        del output
        query = kwargs["query"] if "query" in kwargs else args[0]
        key = kwargs["key"] if "key" in kwargs else args[1]
        seq_len, batch_size, dim = query.shape
        kv_len = key.shape[0]
        num_heads = kwargs["num_heads"] if "num_heads" in kwargs else args[4]
        head_dim = dim // num_heads
        return 4 * batch_size * num_heads * (seq_len / 1e6) * (kv_len / 1e6) * head_dim

    def forward(self, *args, **kwargs):
        return torch.nn.functional.multi_head_attention_forward(*args, **kwargs)


def _custom_openai_residual_attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    return self.mfu_attention_forward(self.attn, x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


def _custom_openai_attention_pool_forward(self, x):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    x = x + self.positional_embedding[:, None, :].to(x.dtype)
    x, _ = self.mfu_attention_forward(
        query=x,
        key=x,
        value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight,
        k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=self.c_proj.weight,
        out_proj_bias=self.c_proj.bias,
        use_separate_proj_weight=True,
        training=self.training,
        need_weights=False,
    )
    return x[0]


class VBenchAestheticRewardModel(BaseRewardModel):
    def __init__(
        self,
        clip_model_name_or_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        frame_batch_size: int = 32,
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchAestheticRewardModel", allow_none=True)
        if frame_batch_size <= 0:
            raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}")
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")
        if clip_model_name_or_path is None:
            raise ValueError("clip_model_name_or_path or pretrained_model_name_or_path must be provided for VBenchAestheticRewardModel")

        try:
            import clip
            from clip.model import AttentionPool2d, ResidualAttentionBlock
        except ImportError as e:
            raise ImportError("clip is required for VBenchAestheticRewardModel") from e

        self.clip_model_name_or_path = clip_model_name_or_path
        self.fps = fps
        self.num_frames = num_frames
        self.frame_batch_size = frame_batch_size
        self.use_norm = use_norm
        self.mean = mean
        self.std = std
        clip_model_name_or_path = os.path.expanduser(self.clip_model_name_or_path)
        self.clip_model, _ = clip.load(clip_model_name_or_path, device="cpu")
        for module in self.clip_model.modules():
            if isinstance(module, ResidualAttentionBlock):
                module.mfu_attention_forward = MultiheadAttentionForward()
                module.attention = types.MethodType(_custom_openai_residual_attention, module)
            elif isinstance(module, AttentionPool2d):
                module.mfu_attention_forward = FunctionalMultiheadAttentionForward()
                module.forward = types.MethodType(_custom_openai_attention_pool_forward, module)
        self.aesthetic_model = nn.Linear(768, 1)
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self._align_runtime_dtypes()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(clip_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            if new_key in {"weight", "bias"}:
                new_key = f"aesthetic_model.{new_key}"
            remapped_state_dict[new_key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)

    def _align_runtime_dtypes(self):
        for module in self.clip_model.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()
        self.aesthetic_model.float()

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._align_runtime_dtypes()
        return module

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchAestheticRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))
        processed_videos = []
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            video = self.to_rgb_uint8_video(tensor, "VBenchAestheticRewardModel")
            video = self.sample_video_frames(
                video,
                source_fps=sample_source_fps,
                fps=self.fps,
                num_frames=self.num_frames,
                model_name="VBenchAestheticRewardModel",
            )
            processed_videos.append(video)
        return processed_videos

    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: Optional[bool] = None,
    ):
        if use_norm is None:
            use_norm = self.use_norm
        self._align_runtime_dtypes()
        videos = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        model_device = next(self.clip_model.parameters()).device
        rewards = []
        for video in videos:
            frame_scores = []
            for start in range(0, video.shape[0], self.frame_batch_size):
                frames = self.image_transform(video[start:start + self.frame_batch_size]).to(device=model_device)
                with torch.no_grad():
                    image_feats = self.clip_model.encode_image(frames).to(torch.float32)
                    image_feats = F.normalize(image_feats, dim=-1, p=2)
                    frame_scores.append(self.aesthetic_model(image_feats).reshape(-1) / 10.0)
            rewards.append(torch.cat(frame_scores).mean())

        reward = torch.stack(rewards).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"AES": reward}

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
