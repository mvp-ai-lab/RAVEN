import os
import sys
import types
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from . import BaseRewardModel
# >>>>>>>>>>>>>> RAVEN adaptation: drop MFU instrumentation import >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_imaging_reward.py:16); MFU
# instrumentation (project.utils.mfu) is not ported to this repo, so the import is removed and
# the CustomFlops mixin below is dropped from the class bases.
#         from project.utils.mfu import CustomFlops
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<

# >>>>>>>>>>>>>> RAVEN adaptation: drop CustomFlops mixin >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_imaging_reward.py:19);
# CustomFlops comes from the unported project.utils.mfu; the wrapper only needs nn.Module.
#         class MatmulForward(nn.Module, CustomFlops):
class MatmulForward(nn.Module):
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
    def tflops(self, args, kwargs, output) -> float:
        del output
        lhs = args[0] if len(args) > 0 else kwargs["lhs"]
        rhs = args[1] if len(args) > 1 else kwargs["rhs"]
        batch_size = lhs.shape[0]
        num_heads = lhs.shape[1]
        q_len = lhs.shape[-2]
        head_dim = lhs.shape[-1]
        kv_len = rhs.shape[-1]
        return 2 * batch_size * num_heads * (q_len / 1e6) * (kv_len / 1e6) * head_dim

    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)

def _custom_musiq_attention_forward(self, x, mask=None):
    B, N, C = x.shape
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    attn = self.qk_matmul(q, k.transpose(-2, -1)) * self.scale
    if mask is not None:
        mask_h = mask.reshape(B, 1, N, 1)
        mask_w = mask.reshape(B, 1, 1, N)
        mask2d = mask_h * mask_w
        attn = attn.masked_fill(mask2d == 0, -1e3)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = self.av_matmul(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.out(x)
    x = self.out_drop(x)
    return x

class VBenchImagingRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        frame_batch_size: int = 8,
        preprocess_mode: str = "longer",
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchImagingRewardModel", allow_none=True)
        if frame_batch_size <= 0:
            raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}")
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")
        if not hasattr(np, "sctypes"):
            np.sctypes = {
                "int": [np.int8, np.int16, np.int32, np.int64],
                "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                "float": [np.float16, np.float32, np.float64],
                "complex": [np.complex64, np.complex128],
                "others": [np.bool_, np.object_, np.str_, np.bytes_],
            }

        # >>>>>>>>>>>>>> timm 0.6 compatibility >>>>>>>>>>>>>>
        # NOTE: importing any pyiqa arch pulls the whole registry, so archs we
        # never use drag in timm 0.9+ module paths: `timm.models._builder` (the
        # renamed `timm.models.helpers`) and the top-level `timm.layers` (moved
        # up from `timm.models.layers`). This repo pins timm 0.6.13 because
        # SANA's own blocks subclass its vision_transformer Attention, so alias
        # the old modules instead of moving the pin. MUSIQ itself never touches
        # timm -- the aliases only keep the unused archs importable.
        import timm.models.helpers
        import timm.models.layers
        sys.modules.setdefault("timm.models._builder", timm.models.helpers)
        sys.modules.setdefault("timm.layers", timm.models.layers)
        # <<<<<<<<<<<<<< end timm 0.6 compatibility <<<<<<<<<<<<<<

        try:
            from pyiqa.archs.musiq_arch import MUSIQ, MultiHeadAttention
        except ImportError as e:
            raise ImportError("pyiqa is required for VBenchImagingRewardModel") from e

        model_path = os.path.expanduser(model_path) if model_path is not None else None
        self.model = MUSIQ(pretrained=False) if model_path is None else MUSIQ(pretrained_model_path=model_path)
        for module in self.model.modules():
            if isinstance(module, MultiHeadAttention):
                module.qk_matmul = MatmulForward()
                module.av_matmul = MatmulForward()
                module.forward = types.MethodType(_custom_musiq_attention_forward, module)
        self.model.eval()
        self.fps = fps
        self.num_frames = num_frames
        self.frame_batch_size = frame_batch_size
        self.preprocess_mode = preprocess_mode
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
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchImagingRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))
        processed_videos = []
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            video = self.to_rgb_uint8_video(tensor, "VBenchImagingRewardModel")
            video = self.sample_video_frames(
                video,
                source_fps=sample_source_fps,
                fps=self.fps,
                num_frames=self.num_frames,
                model_name="VBenchImagingRewardModel",
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
            images = video.float()
            _, _, height, width = images.size()
            if self.preprocess_mode.startswith("shorter"):
                if min(height, width) > 512:
                    scale = 512.0 / min(height, width)
                    images = transforms.Resize((int(scale * height), int(scale * width)), antialias=False)(images)
                    if self.preprocess_mode == "shorter_centercrop":
                        images = transforms.CenterCrop(512)(images)
            elif self.preprocess_mode == "longer":
                if max(height, width) > 512:
                    scale = 512.0 / max(height, width)
                    images = transforms.Resize((int(scale * height), int(scale * width)), antialias=False)(images)
            elif self.preprocess_mode != "None":
                raise ValueError("Please recheck imaging_quality_mode")
            images = images / 255.0

            frame_scores = []
            for start in range(0, images.shape[0], self.frame_batch_size):
                frames = images[start:start + self.frame_batch_size].to(device=model_device, dtype=model_dtype)
                with torch.no_grad():
                    frame_scores.append(self.model(frames).reshape(-1).to(torch.float32))
            rewards.append(torch.cat(frame_scores).mean() / 100.0)

        reward = torch.stack(rewards).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"IMG": reward}

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

# >>>>>>>>>>>>>> RAVEN adaptation: drop __main__ smoke block >>>>>>>>>>>>>>
# NOTE: upstream 原样 (RAVEN: project/models/reward_models/vbench_imaging_reward.py:202-311); the
# smoke block depends on unported project.utils.comm / project.utils.config, so it is removed.
#         if __name__ == "__main__":
#             from decord import VideoReader
#
#             from project.utils import comm
#             from project.utils.config import CfgNode
#
#             parser = argparse.ArgumentParser(description="Distributed VBench imaging first/last-second drift evaluation.")
#             parser.add_argument("--video-dir", required=True)
#             parser.add_argument("--model-path", default="/root/models/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth")
#             parser.add_argument("--seconds", type=float, default=1.0)
#             args = parser.parse_args()
#
#             if args.seconds <= 0:
#                 raise ValueError(f"seconds must be positive, got {args.seconds}")
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
#                     dist.init_process_group(
#                         backend="nccl" if device.type == "cuda" else "gloo",
#                         rank=rank,
#                         world_size=world_size,
#                         timeout=timedelta(minutes=60),
#                     )
#                     dist_initialized = True
#
#                 video_paths = sorted(
#                     os.path.join(args.video_dir, entry.name)
#                     for entry in os.scandir(args.video_dir)
#                     if entry.is_file() and os.path.splitext(entry.name)[1].lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
#                 )
#                 if not video_paths:
#                     raise FileNotFoundError(f"No videos found under {args.video_dir}")
#
#                 model_config = CfgNode.parse_config("configs/models/vbench_imaging.jsonc")
#                 model_config["model_path"] = args.model_path
#                 model = VBenchImagingRewardModel(**{k: v for k, v in model_config.items() if not k.startswith("_")})
#                 if device.type == "cuda":
#                     model.to(device=device, dtype=torch.bfloat16)
#                 else:
#                     model.to(device=device)
#                 model.eval().requires_grad_(False)
#
#                 local_video_paths = video_paths[rank::world_size]
#                 local_results = []
#                 for video_path in tqdm(local_video_paths, disable=local_rank != 0, desc=f"VBench IMG drift node {rank // local_world_size}"):
#                     reader = VideoReader(video_path)
#                     total_frames = len(reader)
#                     source_fps = float(reader.get_avg_fps())
#                     window_frames = max(1, min(total_frames, round(source_fps * args.seconds)))
#                     head_indices = list(range(window_frames))
#                     tail_indices = list(range(total_frames - window_frames, total_frames))
#                     head_frames = reader.get_batch(head_indices).asnumpy()
#                     tail_frames = reader.get_batch(tail_indices).asnumpy()
#                     head_video = torch.from_numpy(head_frames).permute(3, 0, 1, 2).float() / 127.5 - 1.0
#                     tail_video = torch.from_numpy(tail_frames).permute(3, 0, 1, 2).float() / 127.5 - 1.0
#                     with torch.inference_mode():
#                         rewards = model(
#                             video_tensors=[head_video, tail_video],
#                             prompts=["", ""],
#                             source_fps=[source_fps, source_fps],
#                             use_norm=False,
#                         )["IMG"]
#                     head_score = float(rewards[0].detach().cpu().item())
#                     tail_score = float(rewards[1].detach().cpu().item())
#                     local_results.append(
#                         {
#                             "video_path": video_path,
#                             "source_fps": source_fps,
#                             "total_frames": total_frames,
#                             "window_frames": window_frames,
#                             "head_IMG": head_score,
#                             "tail_IMG": tail_score,
#                             "drift_IMG": tail_score - head_score,
#                             "degrade_IMG": max(0.0, head_score - tail_score),
#                         }
#                     )
#
#                 gathered = comm.all_gather_object(local_results)
#                 if local_rank == 0:
#                     results = []
#                     for shard in gathered:
#                         results.extend(shard)
#                     results.sort(key=lambda item: item["video_path"])
#                     summary = {
#                         "node_rank": rank // local_world_size,
#                         "video_dir": args.video_dir,
#                         "seconds": args.seconds,
#                         "video_count": len(results),
#                     }
#                     summary["mean_scores"] = {
#                         "head_IMG": sum(item["head_IMG"] for item in results) / len(results),
#                         "tail_IMG": sum(item["tail_IMG"] for item in results) / len(results),
#                         "drift_IMG": sum(item["drift_IMG"] for item in results) / len(results),
#                         "degrade_IMG": sum(item["degrade_IMG"] for item in results) / len(results),
#                     }
#                     print(json.dumps(summary, ensure_ascii=False, indent=2))
#             finally:
#                 if dist_initialized and dist.is_initialized():
#                     dist.destroy_process_group()
# <<<<<<<<<<<<<< end RAVEN adaptation <<<<<<<<<<<<<<
