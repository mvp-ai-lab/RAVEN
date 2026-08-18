from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseRewardModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def validate_frame_sampling(
        fps: Optional[float],
        num_frames: Optional[int],
        model_name: str,
        allow_none: bool = False,
    ):
        if fps is not None and num_frames is not None:
            raise ValueError(f"fps and num_frames are mutually exclusive for {model_name}")
        if not allow_none and fps is None and num_frames is None:
            raise ValueError(f"fps or num_frames must be provided for {model_name}")

    @staticmethod
    def validate_video_prompt_batch(
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        model_name: str,
    ):
        if len(video_tensors) != len(prompts):
            raise ValueError(
                f"Expected the same number of videos and prompts for {model_name}, got {len(video_tensors)} and {len(prompts)}"
            )

    @staticmethod
    def normalize_source_fps(
        source_fps: Union[float, List[Optional[float]]],
        batch_size: int,
    ) -> List[Optional[float]]:
        if isinstance(source_fps, (int, float)):
            source_fps = [float(source_fps)] * batch_size
        else:
            source_fps = list(source_fps)
        if len(source_fps) != batch_size:
            raise ValueError(f"Expected source_fps to match video batch size {batch_size}, got {len(source_fps)}")
        return source_fps

    @staticmethod
    def to_rgb_uint8_video(
        tensor: torch.Tensor,
        model_name: str,
    ) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Expected video tensor with shape [C, T, H, W] for {model_name}, got {tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3):
            raise ValueError(f"Expected channel dimension C to be 1 or 3 for {model_name}, got {tensor.shape[0]}")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"Expected floating-point video tensor in [-1, 1] for {model_name}, got dtype {tensor.dtype}")
        video_min = tensor.amin().item()
        video_max = tensor.amax().item()
        if video_min < -1.0 or video_max > 1.0:
            raise ValueError(
                f"Expected video tensor range in [-1, 1] for {model_name}, got min={video_min:.4f}, max={video_max:.4f}"
            )
        video = ((tensor.detach().float() + 1.0) * 127.5).round().clamp(0.0, 255.0).to(torch.uint8)
        video = video.permute(1, 0, 2, 3).contiguous()
        if video.shape[1] == 1:
            video = video.repeat(1, 3, 1, 1)
        return video

    @staticmethod
    def sample_video_frames(
        video: torch.Tensor,
        source_fps: Optional[float],
        fps: Optional[float],
        num_frames: Optional[int],
        model_name: str,
    ) -> torch.Tensor:
        if num_frames is not None:
            sample_frames = min(int(num_frames), video.shape[0])
            idx = torch.linspace(0, video.shape[0] - 1, sample_frames, device=video.device).round().long()
            return video[idx]
        if fps is not None:
            if source_fps is None:
                raise ValueError(f"source_fps must be provided when sampling by fps for {model_name}")
            interval = max(1, round(float(source_fps) / float(fps)))
            return video[::interval]
        return video

    @staticmethod
    def collect_sub_reward_models(models: Dict[str, nn.Module]) -> List[Tuple[str, nn.Module]]:
        reward_models = []
        for name in sorted(models.keys()):
            if not name.startswith("reward_model_") or name.endswith("_ema"):
                continue
            model = models[name]
            if model is None:
                continue
            reward_models.append((name, model))
        if len(reward_models) == 0:
            raise ValueError("BaseRewardModel placeholder requires at least one model named reward_model_*")
        return reward_models

    @staticmethod
    def reward_field_prefix(model_name: str) -> str:
        prefix = model_name[len("reward_model_"):]
        if len(prefix) == 0:
            raise ValueError(f"reward_model placeholder child name must have a suffix, got {model_name}")
        return f"{prefix}_"

    @staticmethod
    def merge_reward_outputs(reward_outputs: List[Tuple[str, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        merged = {}
        for model_name, reward_output in reward_outputs:
            if not isinstance(reward_output, dict):
                raise TypeError(f"{model_name} must return a dict, got {type(reward_output)}")
            prefix = BaseRewardModel.reward_field_prefix(model_name)
            for key, value in reward_output.items():
                merged_key = f"{prefix}{key}"
                if merged_key in merged:
                    raise ValueError(f"Duplicate reward field {merged_key} found while merging reward models")
                merged[merged_key] = value
        return merged

    def forward(self, *args, **kwargs):
        raise RuntimeError("BaseRewardModel is a placeholder and should not be called directly")
