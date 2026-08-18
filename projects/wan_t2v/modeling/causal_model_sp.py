"""Ulysses sequence-parallel overrides for the causal Wan model."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.cuda.amp as amp
from torch.nn.attention.flex_attention import BlockMask

from common.distributed.unified_parallel import (
    Gather,
    Slice,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, get_execution_phase

from . import model as wan
from .causal_model import CausalWanModel, CausalWanSelfAttention, NaiveCache, apply_latent_pos_embed
from .checkpointing import maybe_checkpoint


class CausalWanSelfAttentionSP(CausalWanSelfAttention):
    """Causal Wan self-attention with Ulysses and local padded-head caches."""

    def forward(
        self,
        packed_sequence,
        grid_sizes,
        freqs,
        frame_shifts,
        attention_mask,
        q_ranges,
        k_ranges,
        attn_type_map,
        attn_workloads,
        sample_lens,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
        key_value_lens: torch.IntTensor = None,
        packed_query_indexes: Optional[torch.IntTensor] = None,
        packed_past_key_value_indexes: Optional[torch.IntTensor] = None,
    ):
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(
                packed_sequence,
                grid_sizes,
                freqs,
                frame_shifts,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                sample_lens,
                past_key_values,
                update_past_key_values,
                key_value_lens,
                packed_query_indexes,
                packed_past_key_value_indexes,
            )

        local_s = packed_sequence.size(0)
        n, d = self.num_heads, self.head_dim
        up_size = get_unified_parallel_world_size()
        padded_heads = ((n + up_size - 1) // up_size) * up_size
        padded_inner_dim = padded_heads * d

        q = self.norm_q(self.q(packed_sequence))
        k = self.norm_k(self.k(packed_sequence))
        v = self.v(packed_sequence)
        if padded_heads != n:
            head_padding = padded_inner_dim - n * d
            q = torch.cat([q, q.new_zeros(local_s, head_padding)], dim=-1)
            k = torch.cat([k, k.new_zeros(local_s, head_padding)], dim=-1)
            v = torch.cat([v, v.new_zeros(local_s, head_padding)], dim=-1)

        qkv = gather_seq_scatter_heads_qkv(torch.cat([q, k, v], dim=-1), seq_dim=0)
        local_inner_dim = padded_inner_dim // up_size
        q, k, v = qkv.split(local_inner_dim, dim=-1)
        global_s = q.size(0)
        local_heads = padded_heads // up_size
        q = q.view(global_s, local_heads, d)
        k = k.view(global_s, local_heads, d)
        v = v.view(global_s, local_heads, d)

        true_len = sum(math.prod(grid_size) for grid_size in grid_sizes)
        pad_len = global_s - true_len
        q = torch.cat([
            apply_latent_pos_embed(q[:true_len], grid_sizes, freqs, frame_shifts, packed=True),
            q[true_len:],
        ])
        k = torch.cat([
            apply_latent_pos_embed(k[:true_len], grid_sizes, freqs, frame_shifts, packed=True),
            k[true_len:],
        ])

        if past_key_values is not None:
            real_k = k[:true_len]
            real_v = v[:true_len]
            if past_key_values.seq_lens(self.layer_idx) > 0:
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                merged_len = len(packed_query_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(merged_len, local_heads, d)
                merged_value_states = past_value_states.new_zeros(merged_len, local_heads, d)
                merged_key_states[packed_query_indexes] = real_k
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_query_indexes] = real_v
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                real_k, real_v = merged_key_states, merged_value_states

            if update_past_key_values:
                past_key_values.update_kvcache(
                    self.layer_idx,
                    real_k,
                    real_v,
                )

            packed_key_states = torch.cat([real_k, k[true_len:]])
            packed_value_states = torch.cat([real_v, v[true_len:]])
            packed_attn_output = self.flash_attention(
                q,
                packed_key_states,
                packed_value_states,
                q_lens=sample_lens,
                k_lens=key_value_lens,
            )
        else:
            packed_attn_output = self.flex_attention(
                q,
                k,
                v,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                sample_lens,
            )

        packed_attn_output = gather_heads_scatter_seq(
            packed_attn_output.flatten(1), head_dim=1, seq_dim=0
        )
        packed_attn_output = packed_attn_output[:, : n * d]
        return self.o(packed_attn_output)


class CausalWanModelSP(CausalWanModel):
    """Causal Wan model with packed-token sharding and Ulysses self-attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            if not isinstance(block.self_attn, CausalWanSelfAttention):
                raise TypeError(f"Unsupported self-attention type: {type(block.self_attn).__name__}")
            block.self_attn.__class__ = CausalWanSelfAttentionSP

    def forward(
        self,
        x,
        t,
        context,
        packed_position_ids: torch.IntTensor,
        packed_latent_indexes: torch.IntTensor,
        packed_latent_seqlens: torch.IntTensor,
        packed_noisy_latent_relative_indexes: torch.IntTensor,
        packed_noisy_latent_seqlens: torch.IntTensor,
        sample_lens: List[int],
        frame_shifts: List[int],
        attention_mask: Optional[BlockMask] = None,
        q_ranges: Optional[torch.IntTensor] = None,
        k_ranges: Optional[torch.IntTensor] = None,
        attn_type_map: Optional[torch.IntTensor] = None,
        attn_workloads: Optional[List[int]] = None,
        past_key_values_self_attn: Optional[NaiveCache] = None,
        update_past_key_values_self_attn: bool = False,
        past_key_values_cross_attn: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn: bool = False,
        past_key_values_cross_attn_img: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn_img: bool = False,
        clip_fea=None,
        y=None,
        guidance=None,
    ):
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(
                x,
                t,
                context,
                packed_position_ids,
                packed_latent_indexes,
                packed_latent_seqlens,
                packed_noisy_latent_relative_indexes,
                packed_noisy_latent_seqlens,
                sample_lens,
                frame_shifts,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                past_key_values_self_attn,
                update_past_key_values_self_attn,
                past_key_values_cross_attn,
                update_past_key_values_cross_attn,
                past_key_values_cross_attn_img,
                update_past_key_values_cross_attn_img,
                clip_fea,
                y,
                guidance,
            )

        from .pnp_repeat import pnp_repeat

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.int32) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        packed_latent_tokens = torch.cat(x, dim=1)[0]

        true_sequence_len = len(packed_position_ids)
        packed_sequence = packed_latent_tokens.new_zeros(true_sequence_len, self.dim)
        packed_sequence[packed_latent_indexes] = packed_latent_tokens

        sample_lens = torch.tensor(sample_lens, dtype=torch.int32, device=device)
        bsz = len(context) if context is not None and len(context) > 0 else len(x)
        real_sample_lens = sample_lens[:bsz]

        if past_key_values_self_attn is not None:
            if past_key_values_self_attn.seq_len > 0:
                query_len = packed_sequence.shape[0]
                query_lens_tensor = real_sample_lens
                past_key_value_len = sum(past_key_values_self_attn.kvlens)
                past_key_value_lens_tensor = torch.tensor(
                    past_key_values_self_attn.kvlens, dtype=torch.int32, device=device
                )

                key_value_lens_cumsum = torch.cumsum(past_key_value_lens_tensor, dim=0)
                key_value_lens_cumsum_repeat = pnp_repeat(
                    key_value_lens_cumsum, repeats=query_lens_tensor
                )
                packed_query_indexes_self_attn = (
                    torch.arange(query_len, device=device) + key_value_lens_cumsum_repeat
                )

                query_lens_cumsum = torch.cumsum(query_lens_tensor, dim=0)
                query_lens_cumsum = torch.cat(
                    [torch.tensor([0], device=device), query_lens_cumsum[:-1]], dim=0
                )
                query_lens_cumsum_repeat = pnp_repeat(
                    query_lens_cumsum, repeats=past_key_value_lens_tensor
                )
                packed_past_key_value_indexes_self_attn = (
                    torch.arange(past_key_value_len, device=device) + query_lens_cumsum_repeat
                )
                key_value_lens_self_attn = torch.cat([
                    real_sample_lens[i : i + 1] + past_key_values_self_attn.kvlens[i]
                    for i in range(bsz)
                ])

                past_position_ids = pnp_repeat(
                    past_key_values_self_attn.curr_rope, repeats=query_lens_tensor
                )
                packed_position_ids = packed_position_ids + past_position_ids
            else:
                packed_query_indexes_self_attn = None
                packed_past_key_value_indexes_self_attn = None
                key_value_lens_self_attn = real_sample_lens

            if update_past_key_values_self_attn:
                past_key_values_self_attn.update_kvlens(real_sample_lens.tolist())
                position_ids = packed_position_ids.split(real_sample_lens.tolist())
                curr_rope = [position_ids[i][-1] + 1 for i in range(bsz)]
                past_key_values_self_attn.curr_rope = torch.tensor(curr_rope).to(
                    device, non_blocking=True
                )
        else:
            packed_query_indexes_self_attn = None
            packed_past_key_value_indexes_self_attn = None
            key_value_lens_self_attn = None

        def time_emb(t, packed_latent_indexes, noisy_indexes, noisy_seqlens):
            packed_timesteps = t.new_zeros(size=(len(packed_latent_indexes),))
            if len(noisy_indexes) > 0:
                packed_timesteps[noisy_indexes] = pnp_repeat(t, repeats=noisy_seqlens)
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    wan.sinusoidal_embedding_1d(self.freq_dim, packed_timesteps).float()
                )
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            return e, e0

        e, e0 = maybe_checkpoint(
            time_emb,
            t,
            packed_latent_indexes,
            packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens,
            enabled=self.gradient_checkpointing and self.embed_checkpoint_enabled,
        )
        if get_execution_phase() is ExecutionPhase.TRAIN_FORWARD:
            get_running_average_meter().put_scalar("running/time_emb/std", e0.std().item())

        if self.guidance_embeds:
            def guidance_emb(e0, guidance, latent_indexes, noisy_indexes, noisy_seqlens):
                packed_guidance = t.new_zeros(
                    size=(len(latent_indexes),),
                    dtype=guidance.dtype if guidance is not None else torch.float32,
                )
                if len(noisy_indexes) > 0:
                    assert guidance is not None, "guidance embeddings required for guidance_embeds=True"
                    guidance = guidance * 1000
                    packed_guidance[noisy_indexes] = pnp_repeat(guidance, repeats=noisy_seqlens)
                with amp.autocast(dtype=torch.float32):
                    g = self.guidance_embedding(
                        wan.sinusoidal_embedding_1d(self.freq_dim, packed_guidance).float()
                    )
                    g0 = self.guidance_projection(g).unflatten(1, (6, self.dim))
                    if get_execution_phase() is ExecutionPhase.TRAIN_FORWARD:
                        get_running_average_meter().put_scalar(
                            "running/guidance_embeds/std", g0.std().item()
                        )
                    return e0 + g0

            e0 = maybe_checkpoint(
                guidance_emb,
                e0,
                guidance,
                packed_latent_indexes,
                packed_noisy_latent_relative_indexes,
                packed_noisy_latent_seqlens,
                enabled=self.gradient_checkpointing and self.embed_checkpoint_enabled,
            )

        if context is not None and len(context) > 0:
            key_value_lens_cross_attn = torch.tensor(
                [self.text_len for _ in context], dtype=torch.int32, device=device
            )
            context = self.text_embedding(
                torch.cat([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ])
            )
        else:
            context = None
            key_value_lens_cross_attn = None

        (
            key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn,
            packed_past_key_value_indexes_cross_attn,
        ) = self.preprocess_kvcache_cross_attn(
            context,
            key_value_lens_cross_attn,
            past_key_values_cross_attn,
            update_past_key_values_cross_attn,
        )

        up_size = get_unified_parallel_world_size()
        group = get_unified_parallel_group()
        represented_len = max(true_sequence_len, int(sample_lens.sum().item()))
        padded_sequence_len = ((represented_len + up_size - 1) // up_size) * up_size
        pad_len = padded_sequence_len - true_sequence_len
        if pad_len:
            packed_sequence = torch.cat(
                [packed_sequence, packed_sequence.new_zeros(pad_len, self.dim)], dim=0
            )
            e0 = torch.cat([e0, e0.new_zeros(pad_len, 6, self.dim)], dim=0)

        block_sample_lens = real_sample_lens
        if pad_len:
            pad_len_tensor = real_sample_lens.new_tensor([pad_len])
            block_sample_lens = torch.cat([real_sample_lens, pad_len_tensor])
            if key_value_lens_self_attn is not None:
                key_value_lens_self_attn = torch.cat(
                    [key_value_lens_self_attn, pad_len_tensor]
                )
            pad_range = real_sample_lens.new_tensor(
                [[true_sequence_len, padded_sequence_len]]
            )
            if q_ranges is not None:
                q_ranges = torch.cat([q_ranges, pad_range], dim=0)
                k_ranges = torch.cat([k_ranges, pad_range], dim=0)
                attn_type_map = torch.cat(
                    [attn_type_map, attn_type_map.new_zeros(1)], dim=0
                )
                attn_workloads = list(attn_workloads) + [pad_len * pad_len]
            if attention_mask is not None and padded_sequence_len != represented_len:
                # The meta model aligns its padding to the SP world size, so the
                # BlockMask it built is already this length and is consumed as
                # is. Rebuilding it here would bypass the fusion knob that
                # create_sparse_mask owns, and rebuilding means realizing the
                # whole [1, 1, N, N] bool.
                raise ValueError(
                    f"packed BlockMask covers {represented_len} rows but sequence "
                    f"parallelism needs {padded_sequence_len}; the meta model must "
                    f"align sample_lens to the SP world size {up_size}"
                )

        local_sequence_len = padded_sequence_len // up_size
        up_rank = get_unified_parallel_rank()
        local_start = up_rank * local_sequence_len
        local_end = local_start + local_sequence_len
        q_lens = []
        sample_start = 0
        for sample_len in real_sample_lens.tolist():
            sample_end = sample_start + sample_len
            q_lens.append(max(0, min(local_end, sample_end) - max(local_start, sample_start)))
            sample_start = sample_end
        if pad_len:
            q_lens[-1] += max(
                0,
                min(local_end, padded_sequence_len) - max(local_start, true_sequence_len),
            )
        q_lens = torch.tensor(q_lens, dtype=torch.int32, device=device)
        key_value_lens_cross_attn = (q_lens, key_value_lens_cross_attn)

        if self.model_type == "i2v":
            if clip_fea is not None and len(clip_fea) > 0:
                context_clip = self.img_emb(clip_fea)
                key_value_lens_cross_attn_img = torch.tensor(
                    [context_clip.size(1) for _ in context_clip],
                    dtype=torch.int32,
                    device=device,
                )
                context_clip = context_clip.view(-1, context_clip.size(2))
            else:
                context_clip = None
                key_value_lens_cross_attn_img = None

            (
                key_value_lens_cross_attn_img,
                packed_new_key_value_indexes_cross_attn_img,
                packed_past_key_value_indexes_cross_attn_img,
            ) = self.preprocess_kvcache_cross_attn(
                context_clip,
                key_value_lens_cross_attn_img,
                past_key_values_cross_attn_img,
                update_past_key_values_cross_attn_img,
            )
            key_value_lens_cross_attn_img = (q_lens, key_value_lens_cross_attn_img)

            context = [context, context_clip]
            key_value_lens_cross_attn = [
                key_value_lens_cross_attn,
                key_value_lens_cross_attn_img,
            ]
            packed_new_key_value_indexes_cross_attn = [
                packed_new_key_value_indexes_cross_attn,
                packed_new_key_value_indexes_cross_attn_img,
            ]
            packed_past_key_value_indexes_cross_attn = [
                packed_past_key_value_indexes_cross_attn,
                packed_past_key_value_indexes_cross_attn_img,
            ]
            past_key_values_cross_attn = [
                past_key_values_cross_attn,
                past_key_values_cross_attn_img,
            ]
            update_past_key_values_cross_attn = [
                update_past_key_values_cross_attn,
                update_past_key_values_cross_attn_img,
            ]

        packed_sequence = Slice.apply(group, packed_sequence, 0, True)
        e0 = Slice.apply(group, e0, 0, True)
        local_latent_indexes = torch.arange(local_sequence_len, dtype=torch.int32, device=device)

        kwargs = dict(
            e=e0,
            attention_mask=attention_mask,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
            sample_lens=block_sample_lens,
            grid_sizes=grid_sizes.tolist(),
            freqs=self.freqs,
            frame_shifts=frame_shifts,
            context=context,
            past_key_values_self_attn=past_key_values_self_attn,
            update_past_key_values_self_attn=update_past_key_values_self_attn,
            key_value_lens_self_attn=key_value_lens_self_attn,
            packed_query_indexes_self_attn=packed_query_indexes_self_attn,
            packed_past_key_value_indexes_self_attn=packed_past_key_value_indexes_self_attn,
            past_key_values_cross_attn=past_key_values_cross_attn,
            update_past_key_values_cross_attn=update_past_key_values_cross_attn,
            key_value_lens_cross_attn=key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn=packed_new_key_value_indexes_cross_attn,
            packed_past_key_value_indexes_cross_attn=packed_past_key_value_indexes_cross_attn,
            packed_latent_indexes=local_latent_indexes,
        )

        packed_sequence = maybe_checkpoint(
            self.blocks,
            packed_sequence,
            enabled=self.gradient_checkpointing and self.block_checkpoint_enabled,
            gc_step=self.block_checkpoint_step,
            gc_start_idx=self.block_checkpoint_start_idx,
            **kwargs,
        )

        packed_sequence = Gather.apply(group, packed_sequence, 0, True)
        packed_sequence = packed_sequence[:true_sequence_len]
        x = self.head(packed_sequence[packed_latent_indexes], e)
        xs = x.split(packed_latent_seqlens.tolist(), dim=0)
        x = self.unpatchify(xs, grid_sizes)
        return [u.float() for u in x]
