"""Ulysses sequence-parallel overrides for the standard Wan model."""

from __future__ import annotations

import torch
import torch.cuda.amp as amp

from common.distributed.unified_parallel import (
    Gather,
    Slice,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    get_unified_parallel_group,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)

from .checkpointing import maybe_checkpoint
from .model import WanModel, WanSelfAttention, sinusoidal_embedding_1d, rope_apply


class WanSelfAttentionSP(WanSelfAttention):
    """Wan self-attention with a sequence-to-heads Ulysses exchange."""

    def forward(self, x, seq_lens, grid_sizes, freqs):
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(x, seq_lens, grid_sizes, freqs)

        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        up_size = get_unified_parallel_world_size()
        padded_heads = ((n + up_size - 1) // up_size) * up_size
        padded_inner_dim = padded_heads * d

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if padded_heads != n:
            head_padding = padded_inner_dim - n * d
            q = torch.cat([q, q.new_zeros(b, s, head_padding)], dim=-1)
            k = torch.cat([k, k.new_zeros(b, s, head_padding)], dim=-1)
            v = torch.cat([v, v.new_zeros(b, s, head_padding)], dim=-1)

        qkv = gather_seq_scatter_heads_qkv(torch.cat([q, k, v], dim=-1), seq_dim=1)
        local_inner_dim = padded_inner_dim // up_size
        q, k, v = qkv.split(local_inner_dim, dim=-1)
        global_seq_len = q.size(1)
        local_heads = padded_heads // up_size
        q = q.view(b, global_seq_len, local_heads, d)
        k = k.view(b, global_seq_len, local_heads, d)
        v = v.view(b, global_seq_len, local_heads, d)

        x = self.flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        x = gather_heads_scatter_seq(x.flatten(2), head_dim=2, seq_dim=1)
        x = x[..., : n * d]
        return self.o(x)


class WanModelSP(WanModel):
    """Standard Wan model with model-level token sharding and SP attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            if not isinstance(block.self_attn, WanSelfAttention):
                raise TypeError(f"Unsupported self-attention type: {type(block.self_attn).__name__}")
            block.self_attn.__class__ = WanSelfAttentionSP

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        guidance=None,
    ):
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(x, t, context, seq_len, clip_fea, y, guidance)

        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if self.guidance_embeds:
            assert guidance is not None, "guidance embeddings required for guidance_embeds=True"
            guidance = guidance * 1000
            with amp.autocast(dtype=torch.float32):
                g = self.guidance_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, guidance).float()
                )
                g0 = self.guidance_projection(g).unflatten(1, (6, self.dim))
                assert g.dtype == torch.float32 and g0.dtype == torch.float32
                e0 = e0 + g0

        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        if self.model_type == "i2v":
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        up_size = get_unified_parallel_world_size()
        group = get_unified_parallel_group()
        padded_seq_len = ((seq_len + up_size - 1) // up_size) * up_size
        if padded_seq_len != seq_len:
            x = torch.cat([x, x.new_zeros(x.size(0), padded_seq_len - seq_len, x.size(2))], dim=1)
        x = Slice.apply(group, x, 1, True)

        x = maybe_checkpoint(
            self.blocks,
            x,
            enabled=self.gradient_checkpointing and self.block_checkpoint_enabled,
            gc_step=self.block_checkpoint_step,
            gc_start_idx=self.block_checkpoint_start_idx,
            **kwargs,
        )

        x = Gather.apply(group, x, 1, True)
        x = x[:, :seq_len]
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
