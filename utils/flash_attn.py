"""Variable-length attention wrappers with optional kernel acceleration.

Supports packed ``[L, N, C]`` and batched ``[B, L, N, C]`` layouts.
FlashAttention 3 is preferred when available, FlashAttention 2 is used
otherwise, and packed variable-length calls can fall back to scaled dot-product
attention when neither backend is installed. The fallback accepts packed inputs
only and preserves separate query and key sequence boundaries for
self-attention, cross-attention, and cached reads.

The module provides both ``nn.Module`` and functional interfaces and preserves
the input dtype on output.
"""
import os
import warnings

import torch
import torch.nn as nn

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_hopper
    FLASH_ATTN_3_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_3_AVAILABLE", "1")))
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_2_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_2_AVAILABLE", "1")))
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False



# ===== PACKED SDPA FALLBACK BEGIN =====
# FlashAttention kernels require CUDA; this path preserves packed sequence
# boundaries, scaling, causal behavior, and input dtype when FA2/FA3 are absent.
def _sdpa_varlen(q, k, v, *, q_bounds, k_bounds, dropout_p, softmax_scale, causal):
    """Attention over packed [L, N, C] rows, one document per [start, stop).

    Query and key bounds are separate, so this serves cross-attention and cached
    causal reads as well as self-attention. A shared bound list would slice k and
    v with query offsets whenever the layouts differ, including text
    cross-attention (``k_lens = text_lens``) and cached reads
    (``k_lens = q_lens + history``).
    """
    out = torch.empty_like(q)
    pairs = zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
    for q_start, q_stop, k_start, k_stop in pairs:
        if q_start == q_stop:
            continue
        if causal and (q_stop - q_start) != (k_stop - k_start):
            raise ValueError(
                f"causal attention with q_len={q_stop - q_start} != k_len={k_stop - k_start}: "
                "torch's scaled_dot_product_attention aligns the causal triangle TOP-LEFT "
                "when L != S, while FlashAttention 2/3 varlen aligns it BOTTOM-RIGHT, so "
                "this fallback would silently disagree with the kernel it stands in for. "
                "Bottom-right alignment is deliberately not emulated here."
            )
        segment = torch.nn.functional.scaled_dot_product_attention(
            q[q_start:q_stop].transpose(0, 1),
            k[k_start:k_stop].transpose(0, 1),
            v[k_start:k_stop].transpose(0, 1),
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(0, 1)
        out[q_start:q_stop].copy_(segment)
    return out
# ===== PACKED SDPA FALLBACK END =====


class _FlashAttentionVarlen(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        causal = kwargs.get("causal", False)
        _, h, d = output.shape
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if causal:
            min_s = torch.min(seqlens_q, seqlens_k)
            square_part = ((1 + min_s) * min_s) / 2
            full_part = torch.relu(seqlens_q - seqlens_k) * min_s
            valid = (full_part / 1e12 + square_part / 1e12).sum()
        else:
            valid = ((seqlens_q / 1e6) * (seqlens_k / 1e6)).sum()
        return h * (4 * d * valid)

    def forward(self, *args, **kwargs):
        version = kwargs.pop("version", None)
        # apply attention
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            return flash_attn_varlen_func_hopper(*args, **kwargs)
        else:
            assert FLASH_ATTN_2_AVAILABLE
            return flash_attn_varlen_func(*args, **kwargs)


class FlashAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_attn_impl = _FlashAttentionVarlen()

    def forward(
        self,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        """
        q:              [B, Lq, Nq, C1] or packed [L, Nq, C1].
        k:              [B, Lk, Nk, C1] or packed [L, Nk, C1].
        v:              [B, Lk, Nk, C2] or packed [L, Nk, C2]. Nq must be divisible by Nk.
        q_lens:         [B].
        k_lens:         [B].
        dropout_p:      float. Dropout probability.
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        causal:         bool. Whether to apply causal attention mask.
        window_size:    (left right). If not (-1, -1), apply sliding window local attention.
        deterministic:  bool. If True, slightly slower and uses more memory.
        dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
        """
        # ===== PACKED SDPA DISPATCH BEGIN =====
        # Without FA2/FA3, dispatch packed attention through ``_sdpa_varlen``
        # while preserving sequence boundaries, scale, causal mode, and dtype.
        if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            assert q.ndim == 3 and q_scale is None, \
                "the SDPA fallback only serves packed [L, N, C] attention without q_scale"
            assert q_lens is not None, "q_lens is required for packed attention"
            k_lens_effective = q_lens if k_lens is None else k_lens
            if len(k_lens_effective) != len(q_lens):
                raise ValueError(
                    f"q_lens has {len(q_lens)} documents but k_lens has "
                    f"{len(k_lens_effective)}; they describe the same packing"
                )
            q_bounds = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0).tolist()
            k_bounds = (
                torch.cat([k_lens_effective.new_zeros([1]), k_lens_effective])
                .cumsum(0)
                .tolist()
            )
            return _sdpa_varlen(
                q, k, v,
                q_bounds=q_bounds,
                k_bounds=k_bounds,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        # ===== PACKED SDPA DISPATCH END =====
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        # params
        # b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        out_dtype = q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # preprocess query
        q_is_packed = q.ndim == 3
        if q_is_packed:  # packed
            assert q_lens is not None, "q_lens must be provided when q is packed."
            b = len(q_lens)
            # Move the lengths to the host before reducing. Python's max()
            # iterates a CUDA tensor element by element and every comparison
            # forces a device->host sync, so a packed call cost B-1 pipeline
            # drains -- ~23% of the training step on a 16-layer backbone under
            # gradient checkpointing. One small D2H copy replaces all of them.
            lq = int(q_lens.cpu().max())
            q = half(q)
        else:
            b, lq = q.size(0), q.size(1)
            if q_lens is None:
                q = half(q.flatten(0, 1))
                q_lens = torch.tensor(
                    [lq] * b, dtype=torch.int32).to(
                        device=q.device, non_blocking=True)
            else:
                q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        # q_lens = torch.tensor(q_lens, dtype=torch.int32, device=q.device)
        # out_dtype = q.dtype

        # preprocess key, value
        kv_is_packed = k.ndim == 3
        if kv_is_packed:
            assert k_lens is not None, "k_lens must be provided when k is packed."
            lk = int(k_lens.cpu().max())
            k = half(k)
            v = half(v)
        else:
            lk = k.size(1)
            if k_lens is None:
                k = half(k.flatten(0, 1))
                v = half(v.flatten(0, 1))
                k_lens = torch.tensor(
                    [lk] * b, dtype=torch.int32).to(
                        device=k.device, non_blocking=True)
            else:
                k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
                v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
            warnings.warn(
                'Flash attention 3 is not available, use flash attention 2 instead.'
            )

        # apply attention
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            # x = flash_attn_interface.flash_attn_varlen_func(
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)
            x = self._flash_attn_impl(version=3, **kwargs)
        else:
            assert FLASH_ATTN_2_AVAILABLE
            # x = flash_attn.flash_attn_varlen_func(
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic)
            x = self._flash_attn_impl(version=2, **kwargs)

        # output
        if q_is_packed:
            return x.type(out_dtype)
        else:
            return x.unflatten(0, (b, lq)).type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_varlen_func_hopper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
