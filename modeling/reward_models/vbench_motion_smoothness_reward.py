from typing import List, Optional, Union

import numpy as np
import torch

from . import BaseRewardModel
from .vbench_amt.model import Model as AMTSModel
from .vbench_raft.core.utils_core.utils import InputPadder


class VBenchMotionSmoothnessRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchMotionSmoothnessRewardModel", allow_none=True)
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")

        self.model = AMTSModel()
        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt)
        self.model.eval()

        self.fps = fps
        self.num_frames = num_frames
        self.use_norm = use_norm
        self.mean = mean
        self.std = std
        self.niters = 1
        self.anchor_resolution = 1024 * 512
        self.anchor_memory = 1500 * 1024**2
        self.anchor_memory_bias = 2500 * 1024**2
        self.register_buffer("embt", torch.tensor(0.5).view(1, 1, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        kwargs.pop("model_path", None)
        return cls(model_path=pretrained_model_name_or_path, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            if not new_key.startswith("model."):
                new_key = f"model.{new_key}"
            remapped_state_dict[new_key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchMotionSmoothnessRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))
        processed_videos = []
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            video = self.to_rgb_uint8_video(tensor, "VBenchMotionSmoothnessRewardModel")
            video = self.sample_video_frames(
                video,
                source_fps=sample_source_fps,
                fps=self.fps,
                num_frames=self.num_frames,
                model_name="VBenchMotionSmoothnessRewardModel",
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
        videos = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        rewards = []
        for video in videos:
            frames = video.float().div(255.0).to(device=model_device, dtype=model_dtype)
            inputs = [frames[i:i + 1] for i in range(0, frames.shape[0], 2)]
            if len(inputs) <= 1:
                rewards.append(torch.zeros((), device=model_device, dtype=torch.float32))
                continue

            targets = video[1::2].to(device=model_device)
            h, w = inputs[0].shape[-2:]
            if model_device.type == "cuda":
                vram_avail = torch.cuda.get_device_properties(model_device).total_memory
                scale = self.anchor_resolution / (h * w)
                scale = scale * np.sqrt(max(float(vram_avail - self.anchor_memory_bias), 1.0) / float(self.anchor_memory))
                scale = min(scale, 1.0)
                scale = max(scale, 1e-4)
                scale = 1.0 / np.floor(1.0 / np.sqrt(scale) * 16.0) * 16.0
            else:
                scale = 1.0

            padding = int(16 / scale)
            padder = InputPadder(inputs[0].shape, padding)
            inputs = padder.pad(*inputs)
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt.to(device=model_device, dtype=model_dtype), scale_factor=scale, eval=True)["imgt_pred"]
                outputs += [imgt_pred, in_1]

            outputs = [padder.unpad(out) for out in outputs]
            interpolates = torch.cat(outputs[1::2], dim=0).to(torch.float32)
            pair_count = min(interpolates.shape[0], targets.shape[0])
            if pair_count == 0:
                rewards.append(torch.zeros((), device=model_device, dtype=torch.float32))
                continue
            interpolates = (interpolates[:pair_count] * 255.0).clamp(0.0, 255.0).to(torch.uint8)
            diff = (targets[:pair_count].to(torch.float32) - interpolates.to(torch.float32)).abs().mean(dim=(1, 2, 3)) / 255.0
            rewards.append((1.0 - diff).mean())

        reward = torch.stack(rewards).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"MS": reward}

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
