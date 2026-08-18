# SPDX-License-Identifier: Apache-2.0
"""Cross-attention discriminator head for MiniMax H3 DMD2 training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from common.distributed.unified_parallel import (
    Gather,
    SeqAllToAll,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from utils.flash_attn import FlashAttention

from .checkpointing import gradient_checkpointing
from .transformer.config import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
)
from .transformer.model import (
    _BF16_DTYPE,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MiniMaxH3AdalnProj,
    MiniMaxH3MLP,
    RowParallelLinear,
    _apply_qk_norm,
    _modulate_gate,
    _modulate_scale_shift,
    _norm,
)

_MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION = FlashAttention()



def _ulysses_scatter_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    up_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T_local, n, d] row shards -> [T_global, n/up_size, d] head shards."""
    n = k.shape[1]
    if n % up_size:
        raise ValueError(
            f"attention heads {n} must be divisible by the unified-parallel "
            f"world size {up_size}"
        )
    if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % up_size:
        raise ValueError(
            "MiniMax H3 packed sequence alignment "
            f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} must be divisible by "
            f"unified-parallel world size {up_size}"
        )
    group = get_unified_parallel_group()
    assert group is not None
    return (
        SeqAllToAll.apply(group, k, 1, 0, False),
        SeqAllToAll.apply(group, v, 1, 0, False),
    )


class MiniMaxH3DiscriminatorCrossAttention(nn.Module):
    """H3-width cross-attention from replicated learned queries to packed tap rows."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: Any = None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(
            arch.hidden_size,
            self.inner_dim,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = MergedColumnParallelLinear(
            arch.hidden_size,
            [self.inner_dim, self.inner_dim],
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )
        self.q_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, generator=generator)
        nn.init.xavier_uniform_(self.kv_proj.weight, generator=generator)
        nn.init.xavier_uniform_(self.out_proj.weight, generator=generator)
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.k_norm.weight)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Attend packed ``query`` documents to the matching packed ``kv`` documents."""
        if q_lens.numel() != k_lens.numel():
            raise ValueError(
                f"q_lens has {q_lens.numel()} documents but k_lens has "
                f"{k_lens.numel()}"
            )

        q_total = query.shape[0]
        k_total = kv.shape[0]
        q, _ = self.q_proj(query)
        kv, _ = self.kv_proj(kv)
        k, v = kv.split(self.inner_dim, dim=-1)
        q = q.view(q_total, self.num_heads, self.head_dim)
        k = k.view(k_total, self.num_heads, self.head_dim)
        v = v.view(k_total, self.num_heads, self.head_dim)
        q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)

        up_size = (
            get_unified_parallel_world_size()
            if is_unified_parallel_initialized()
            else 1
        )
        if up_size > 1:
            k, v = _ulysses_scatter_kv(k, v, up_size)
            local_heads = self.num_heads // up_size
            head_start = get_unified_parallel_rank() * local_heads
            q = q[:, head_start : head_start + local_heads]

        out = _MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION(
            q,
            k,
            v,
            q_lens=q_lens.to(dtype=torch.int32),
            k_lens=k_lens.to(dtype=torch.int32),
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
        )
        if up_size > 1:
            group = get_unified_parallel_group()
            assert group is not None
            out = Gather.apply(group, out, 1, True)

        out = out.reshape(q_total, self.inner_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3DiscriminatorBlock(nn.Module):
    """One AdaLN-Zero H3 block with learned-query cross-attention and no RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: Any = None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.kv_norm = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3DiscriminatorCrossAttention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")
        # Every discriminator row is a query token, so unlike the trunk block
        # (text/video/audio) there is only one modality to modulate.
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            6 * arch.hidden_size,
            quant_config,
            prefix=f"{prefix}.adaln_proj",
            expand_ratio=6,
            modality_num=1,
        )
        self._enable_gradient_checkpointing = False
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        self.attn.reset_parameters(generator=generator)
        nn.init.xavier_uniform_(self.mlp.fc1.weight, generator=generator)
        nn.init.xavier_uniform_(self.mlp.fc2.weight, generator=generator)
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.kv_norm.weight)
        nn.init.ones_(self.norm2.weight)
        # The feature path must be LIVE at init. With the gates ALSO zero
        # (AdaLN-Zero), every block outputs exactly its residual -- the learned
        # queries, which carry no sample information -- so the head input is
        # bit-identical for the real and the fake pass, the +-sigmoid(0)
        # grad_outs cancel exactly on the zeroed logit weight, and that zero
        # weight blocks every upstream gradient: an exact fixed point (trial
        # 15446442 sat at logits == 0.0 for 80 steps; reproduced in a toy
        # simulation of this block's dataflow). LTX2 avoids it by zeroing only
        # the output layer of an otherwise randomly-initialized conv stack.
        # Same principle here: weight = 0 keeps the start timestep-independent
        # and fifty-fifty, and the bias opens the gates (chunk(6) order is
        # shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp),
        # so the block starts as a plain pre-norm cross-attention block and the
        # timestep modulation ramps in from the zero weight.
        hidden = self.adaln_proj.hidden_size
        nn.init.zeros_(self.adaln_proj.linear.weight)
        with torch.no_grad():
            bias = self.adaln_proj.linear.bias
            bias.zero_()
            bias[2 * hidden : 3 * hidden].fill_(1.0)
            bias[5 * hidden : 6 * hidden].fill_(1.0)

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self._enable_gradient_checkpointing = enable

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
    ) -> torch.Tensor:
        return gradient_checkpointing(
            self._forward,
            query,
            kv,
            use_reentrant=False,
            enabled=self._enable_gradient_checkpointing and self.training and torch.is_grad_enabled(),
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            q_lens=q_lens,
            k_lens=k_lens,
        )

    def _forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
    ) -> torch.Tensor:
        adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaln_params
        )

        residual = query
        h = self.norm1(query)
        h = _modulate_scale_shift(
            h,
            shift_msa,
            scale_msa,
            combined_indices,
            dtype=_BF16_DTYPE,
        )
        h = self.attn(
            h,
            self.kv_norm(kv),
            q_lens=q_lens,
            k_lens=k_lens,
        )
        query = _modulate_gate(
            residual,
            gate_msa,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
        )

        residual = query
        h = self.norm2(query)
        h = _modulate_scale_shift(
            h,
            shift_mlp,
            scale_mlp,
            combined_indices,
            dtype=_BF16_DTYPE,
        )
        h = self.mlp(h)
        return _modulate_gate(
            residual,
            gate_mlp,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
        )


class MiniMaxH3DMD2Discriminator(nn.Module):
    """Learned-query discriminator returning one scalar per sample per tap."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        num_queries: int,
        num_taps: int = 4,
        quant_config: Any = None,
        prefix: str = "discriminator",
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        if num_taps <= 0:
            raise ValueError(f"num_taps must be positive, got {num_taps}")
        self.hidden_size = arch.hidden_size
        self.num_queries = num_queries
        self.num_taps = num_taps
        self.queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        num_queries,
                        arch.hidden_size,
                        dtype=_BF16_DTYPE,
                    )
                )
                for _ in range(self.num_taps)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiscriminatorBlock(
                    arch,
                    quant_config,
                    prefix=f"{prefix}.blocks.{index}",
                )
                for index in range(self.num_taps)
            ]
        )
        head_dim = self.num_queries * arch.hidden_size
        self.final_norm = nn.ModuleList(
            [
                nn.LayerNorm(
                    head_dim,
                    eps=arch.final_norm_eps,
                    dtype=_BF16_DTYPE,
                )
                for _ in range(self.num_taps)
            ]
        )
        self.logit = nn.ModuleList(
            [
                nn.Linear(
                    head_dim,
                    1,
                    dtype=_BF16_DTYPE,
                )
                for _ in range(self.num_taps)
            ]
        )
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Signature matches minimax_h3's other models, which is what NativeGradientCheckpointing drives."""
        for block in self.blocks:
            block.set_gradient_checkpointing(enabled)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        for block in self.blocks:
            block.reset_parameters(generator=generator)
        for query in self.queries:
            nn.init.normal_(query, mean=0.0, std=0.02, generator=generator)
        for final_norm, logit in zip(self.final_norm, self.logit, strict=True):
            nn.init.ones_(final_norm.weight)
            nn.init.zeros_(final_norm.bias)
            # Zero, not Linear's default kaiming: every tap must start at logit
            # 0. Then softplus(0) == softplus(-0) == log 2, so an untrained
            # discriminator scores real and fake identically, the FAKE objective
            # starts with no spurious signal, and the generator's adversarial
            # term starts at zero instead of at whatever a random 21504 -> 1
            # projection happens to say. The head still learns from step 0 --
            # a zeroed weight has zero gradient into its INPUT, not into itself
            # (grad_w = grad_out * input) -- but ONLY because the block gates
            # above start OPEN, so the real and fake passes feed it different
            # inputs; with identical inputs the +-0.5 grad_outs cancel exactly
            # and this layer never leaves zero (see the block init comment).
            nn.init.zeros_(logit.weight)
            if logit.bias is not None:
                nn.init.zeros_(logit.bias)

    def forward(
        self,
        tap_hidden_states: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped ``[B, num_taps]`` for packed tap hidden tensors."""
        if len(tap_hidden_states) != self.num_taps:
            raise ValueError(
                "MiniMax H3 discriminator requires exactly "
                f"{self.num_taps} taps, got "
                f"{len(tap_hidden_states)}"
            )
        num_documents = k_lens.numel()
        if live_documents.numel() != num_documents:
            raise ValueError(
                f"live_documents has {live_documents.numel()} entries but k_lens "
                f"describes {num_documents} documents"
            )
        batch_size = adaln_input.shape[0]

        q_lens = torch.full_like(k_lens, self.num_queries, dtype=torch.int32)
        if bool(((k_lens == 0) & (q_lens > 0)).any()):
            raise ValueError("attention documents with queries must have at least one key")
        document_sample_indices = live_documents.to(torch.long).cumsum(dim=0) - 1
        combined_indices = document_sample_indices.repeat_interleave(self.num_queries)
        logits = []
        for query, block, final_norm, logit, tap_hidden in zip(
            self.queries,
            self.blocks,
            self.final_norm,
            self.logit,
            tap_hidden_states,
            strict=True,
        ):
            packed_query = (
                query.unsqueeze(0)
                .expand(num_documents, -1, -1)
                .reshape(num_documents * self.num_queries, self.hidden_size)
            )
            hidden = block(
                packed_query,
                tap_hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                q_lens=q_lens,
                k_lens=k_lens,
            )
            live_hidden = hidden.view(
                num_documents, self.num_queries, self.hidden_size
            )[live_documents]
            if live_hidden.shape[0] != batch_size:
                raise ValueError(
                    f"live_documents selects {live_hidden.shape[0]} samples but "
                    f"adaln_input has {batch_size} rows"
                )
            logits.append(logit(final_norm(live_hidden.flatten(1))).squeeze(-1))

        return torch.stack(logits, dim=1)


__all__ = [
    "MiniMaxH3DMD2Discriminator",
    "MiniMaxH3DiscriminatorBlock",
    "MiniMaxH3DiscriminatorCrossAttention",
]
