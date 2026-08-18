import types
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn

from . import BaseRewardModel
from .vbench_raft.core.raft import RAFT
from .vbench_raft.core.utils_core.utils import InputPadder


class VBenchRAFTRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        small: bool = False,
        mixed_precision: bool = False,
        alternate_corr: bool = False,
        flow_iters: int = 20,
        topk_ratio: float = 0.05,
        pair_batch_size: int = 8,
        aggregate: Literal["mean", "max"] = "mean",
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchRAFTRewardModel")
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        if pair_batch_size <= 0:
            raise ValueError(f"pair_batch_size must be positive, got {pair_batch_size}")
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")

        args = types.SimpleNamespace(
            model=model_path,
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
        )
        self.model = RAFT(args)
        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu")
            self.load_state_dict(ckpt)
        self.model.eval()

        self.fps = fps
        self.num_frames = num_frames
        self.flow_iters = flow_iters
        self.topk_ratio = topk_ratio
        self.pair_batch_size = pair_batch_size
        self.aggregate = aggregate
        self.use_norm = use_norm
        self.mean = mean
        self.std = std

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(model_path=pretrained_model_name_or_path, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
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
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchRAFTRewardModel")
        del prompts
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))

        processed_videos = []
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            if tensor.ndim != 4:
                raise ValueError(f"Expected video tensor with shape [C, T, H, W], got {tuple(tensor.shape)}")
            if tensor.shape[0] not in (1, 3):
                raise ValueError(f"Expected channel dimension C to be 1 or 3, got {tensor.shape[0]}")
            if sample_source_fps is None and self.num_frames is None:
                raise ValueError("source_fps must be provided when sampling by fps for VBenchRAFTRewardModel")

            video = ((tensor.detach().float() + 1.0) * 127.5).round().clamp(0.0, 255.0)
            video = video.permute(1, 0, 2, 3).contiguous()
            if video.shape[1] == 1:
                video = video.repeat(1, 3, 1, 1)

            if self.num_frames is not None:
                sample_frames = min(int(self.num_frames), video.shape[0])
                idx = torch.linspace(0, video.shape[0] - 1, sample_frames, device=video.device).round().long()
                video = video[idx]
            else:
                interval = max(1, round(float(sample_source_fps) / float(self.fps)))
                video = video[::interval]

            processed_videos.append(video.to(device=device, dtype=dtype))
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
        scores = []
        overall_scores = []
        for video in videos:
            if video.shape[0] < 2:
                scores.append(torch.zeros((), device=video.device, dtype=torch.float32))
                overall_scores.append(torch.zeros((), device=video.device, dtype=torch.float32))
                continue

            pair_scores = []
            for start in range(0, video.shape[0] - 1, self.pair_batch_size):
                image1 = video[start:start + self.pair_batch_size]
                image2 = video[start + 1:start + 1 + self.pair_batch_size]
                if image1.shape[0] == 0 or image2.shape[0] == 0:
                    continue
                pair_count = min(image1.shape[0], image2.shape[0])
                image1 = image1[:pair_count]
                image2 = image2[:pair_count]

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=self.flow_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).float()
                rad = torch.linalg.vector_norm(flow_up, dim=1)
                topk = max(1, int(rad.shape[-2] * rad.shape[-1] * self.topk_ratio))
                pair_scores.append(rad.flatten(1).topk(topk, dim=1).values.mean(dim=1))

            pair_scores = torch.cat(pair_scores)
            threshold = 6.0 * (min(video.shape[-2:]) / 256.0)
            count_num = round(4 * (float(video.shape[0]) / 16.0))
            if self.aggregate == "mean":
                score = pair_scores.mean()
            elif self.aggregate == "max":
                score = pair_scores.max()
            else:
                raise ValueError(f"Unsupported aggregate: {self.aggregate}")
            scores.append(score)
            overall_scores.append((pair_scores > threshold).sum().ge(count_num).to(dtype=torch.float32))

        reward = torch.stack(scores).float()
        overall = torch.stack(overall_scores).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"RAFT": reward, "Overall": overall}

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
