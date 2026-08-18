"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _repeat_interleave_forward_kernel_dim_0(
    input,
    output,
    repeats_cumsum,
    num_rows_in,
    num_rows_out,
    N: tl.constexpr,
    STRIDE_IN_M: tl.constexpr,
    STRIDE_IN_N: tl.constexpr,
    STRIDE_OUT_M: tl.constexpr,
    STRIDE_OUT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    pid_m = tl.program_id(axis=0).to(tl.uint64)  # m

    repeat_start = tl.load(repeats_cumsum + pid_m - 1, mask=pid_m > 0, other=0)
    repeat_end = tl.load(repeats_cumsum + pid_m)
    repeat = (repeat_end - repeat_start).to(tl.uint64)

    block_idx = tl.program_id(axis=1)

    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.device_assert(pid_m < num_rows_in, "Input OOB")
    x = load_with_pred_1d(input + pid_m * STRIDE_IN_M + n * STRIDE_IN_N, N_ALIGNED, mask=n < N, other=0)

    for i in tl.range(repeat):
        tl.device_assert(repeat_start + i < num_rows_out, "Output OOB")
        y = output + (repeat_start + i) * STRIDE_OUT_M + n * STRIDE_OUT_N
        store_with_pred_1d(y, x, N_ALIGNED, mask=n < N)


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _repeat_interleave_forward_kernel_dim_1(
    input,
    output,
    repeats_cumsum,
    num_rows_in,
    num_rows_out,
    N: tl.constexpr,
    STRIDE_IN_G: tl.constexpr,
    STRIDE_IN_M: tl.constexpr,
    STRIDE_IN_N: tl.constexpr,
    STRIDE_OUT_G: tl.constexpr,
    STRIDE_OUT_M: tl.constexpr,
    STRIDE_OUT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid % num_rows_in
    group = pid // num_rows_in

    repeat_start = tl.load(repeats_cumsum + pid_m - 1, mask=pid_m > 0, other=0)
    repeat_end = tl.load(repeats_cumsum + pid_m)
    repeat = (repeat_end - repeat_start).to(tl.uint64)

    block_idx = tl.program_id(axis=1)

    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.device_assert(pid_m < num_rows_in, "Input OOB")
    x = load_with_pred_1d(
        input + group * STRIDE_IN_G + pid_m * STRIDE_IN_M + n * STRIDE_IN_N, N_ALIGNED, mask=n < N, other=0
    )

    for i in tl.range(repeat):
        tl.device_assert(repeat_start + i < num_rows_out, "Output OOB")
        y = output + group * STRIDE_OUT_G + (repeat_start + i) * STRIDE_OUT_M + n * STRIDE_OUT_N
        store_with_pred_1d(y, x, N_ALIGNED, mask=n < N)


def _repeat_interleave_forward(
    input: torch.Tensor,
    repeats: torch.Tensor,
    repeats_cumsum: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
    output_size: Optional[torch.Size] = None,
):
    """
    Parameters
    input (Tensor): the input tensor.
    repeats (Tensor or int): The number of repetitions for each element. repeats is
            broadcasted to fit the shape of the given axis.
    dim (int, optional): The dimension along which to repeat values. By default, use
            the flattened input array, and return a flat output array.

    Keyword Arguments
    output_size (int, optional): Total output size for the given axis ( e.g. sum of repeats).
            If given, it will avoid stream synchronization needed to calculate output shape
            of the tensor.

    Returns
    Repeated tensor which has the same shape as input, except along the given axis.

    Return type
    Tensor
    """

    assert len(input.shape) == 3, len(input.shape)

    if dim is None:
        dim = 0
    assert dim == 0 or dim == 1, dim

    if output_size is None:
        output_size = torch.sum(repeats).item()

    if repeats_cumsum is None:
        repeats_cumsum = torch.cumsum(repeats, dim=0)

    shape = list(input.shape)
    if dim == 0:
        shape[0] = output_size
    elif dim == 1:
        shape[1] = output_size
    output = torch.empty(shape, dtype=input.dtype, device=input.device, requires_grad=False)

    input_shape = input.shape
    output_shape = output.shape

    if dim == 0:
        input = input.view(input.shape[0], -1)
        output = output.view(output.shape[0], -1)

        M, N = input.shape
        grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa

        _repeat_interleave_forward_kernel_dim_0[grid](
            input=input,
            output=output,
            repeats_cumsum=repeats_cumsum,
            num_rows_in=input.shape[0],
            num_rows_out=output.shape[0],
            N=N,
            STRIDE_IN_M=input.stride(0),
            STRIDE_IN_N=input.stride(1),
            STRIDE_OUT_M=output.stride(0),
            STRIDE_OUT_N=output.stride(1),
            BLOCK_N=1024,
        )

        input = input.view(input_shape)
        output = output.view(output_shape)

    elif dim == 1:
        G, M, N = input.shape
        grid = lambda meta: (G * M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa

        _repeat_interleave_forward_kernel_dim_1[grid](
            input=input,
            output=output,
            repeats_cumsum=repeats_cumsum,
            num_rows_in=M,
            num_rows_out=output.shape[1],
            N=N,
            STRIDE_IN_G=input.stride(0),
            STRIDE_IN_M=input.stride(1),
            STRIDE_IN_N=input.stride(2),
            STRIDE_OUT_G=output.stride(0),
            STRIDE_OUT_M=output.stride(1),
            STRIDE_OUT_N=output.stride(2),
            BLOCK_N=1024,
        )

    return output


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _repeat_interleave_backward_kernel_dim_0(
    grad,
    output,
    repeats_cumsum,
    num_rows_in,
    num_rows_out,
    N: tl.constexpr,
    STRIDE_IN_M: tl.constexpr,
    STRIDE_IN_N: tl.constexpr,
    STRIDE_OUT_M: tl.constexpr,
    STRIDE_OUT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    pid_m = tl.program_id(axis=0).to(tl.uint64)  # m

    repeat_start = tl.load(repeats_cumsum + pid_m - 1, mask=pid_m > 0, other=0)
    repeat_end = tl.load(repeats_cumsum + pid_m)
    repeat = (repeat_end - repeat_start).to(tl.uint64)

    block_idx = tl.program_id(axis=1)

    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    y = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.range(repeat):
        tl.device_assert(repeat_start + i < num_rows_in, "Input OOB")
        x = load_with_pred_1d(grad + (repeat_start + i) * STRIDE_IN_M + n * STRIDE_IN_N, N_ALIGNED, mask=n < N, other=0)
        y += x

    # save one line
    tl.device_assert(pid_m < num_rows_out, "Output OOB")
    output = output + pid_m * STRIDE_OUT_M + n * STRIDE_OUT_N  # noqa
    store_with_pred_1d(output, y, N_ALIGNED, mask=n < N)


@triton.heuristics(values={"N_ALIGNED": lambda args: args["N"] % args["BLOCK_N"] == 0})
@triton.jit
def _repeat_interleave_backward_kernel_dim_1(
    grad,
    output,
    repeats_cumsum,
    num_rows_in,
    num_rows_out,
    N: tl.constexpr,
    STRIDE_IN_G: tl.constexpr,
    STRIDE_IN_M: tl.constexpr,
    STRIDE_IN_N: tl.constexpr,
    STRIDE_OUT_G: tl.constexpr,
    STRIDE_OUT_M: tl.constexpr,
    STRIDE_OUT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid % num_rows_out
    group = pid // num_rows_out

    repeat_start = tl.load(repeats_cumsum + pid_m - 1, mask=pid_m > 0, other=0)
    repeat_end = tl.load(repeats_cumsum + pid_m)
    repeat = (repeat_end - repeat_start).to(tl.uint64)

    block_idx = tl.program_id(axis=1)

    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    y = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.range(repeat):
        tl.device_assert(repeat_start + i < num_rows_in, "Input OOB")
        in_ptr = grad + group * STRIDE_IN_G + (repeat_start + i) * STRIDE_IN_M + n * STRIDE_IN_N
        x = load_with_pred_1d(in_ptr, N_ALIGNED, mask=n < N, other=0)
        y += x

    # save one line
    tl.device_assert(pid_m < num_rows_out, "Output OOB")
    out_ptr = output + group * STRIDE_OUT_G + pid_m * STRIDE_OUT_M + n * STRIDE_OUT_N
    store_with_pred_1d(out_ptr, y, N_ALIGNED, mask=n < N)


def _repeat_interleave_backward(
    grad: torch.Tensor,
    input: torch.Tensor,
    repeats: torch.Tensor,
    repeats_cumsum: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
):
    assert len(input.shape) == 3, len(input.shape)

    if dim is None:
        dim = 0
    assert dim == 0 or dim == 1, dim

    if repeats_cumsum is None:
        repeats_cumsum = torch.cumsum(repeats, dim=0)

    output = torch.zeros_like(input)

    if dim == 0:
        grad_shape = grad.shape
        output_shape = output.shape

        grad = grad.view(grad.shape[0], -1)
        output = output.view(output.shape[0], -1)

        M, N = output.shape
        grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa

        _repeat_interleave_backward_kernel_dim_0[grid](
            grad=grad,
            output=output,
            repeats_cumsum=repeats_cumsum,
            num_rows_in=grad_shape[0],
            num_rows_out=M,
            N=N,
            STRIDE_IN_M=grad.stride(0),
            STRIDE_IN_N=grad.stride(1),
            STRIDE_OUT_M=output.stride(0),
            STRIDE_OUT_N=output.stride(1),
            BLOCK_N=1024,
        )

        output = output.view(output_shape)

    elif dim == 1:
        G, M, N = output.shape
        grid = lambda meta: (G * M, triton.cdiv(N, meta["BLOCK_N"]))  # noqa

        _repeat_interleave_backward_kernel_dim_1[grid](
            grad=grad,
            output=output,
            repeats_cumsum=repeats_cumsum,
            num_rows_in=grad.shape[1],
            num_rows_out=output.shape[1],
            N=N,
            STRIDE_IN_G=grad.stride(0),
            STRIDE_IN_M=grad.stride(1),
            STRIDE_IN_N=grad.stride(2),
            STRIDE_OUT_G=output.stride(0),
            STRIDE_OUT_M=output.stride(1),
            STRIDE_OUT_N=output.stride(2),
            BLOCK_N=1024,
        )

    return output


class RepeatInterleave(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        repeats: torch.Tensor,
        dim: Optional[int] = None,
        output_size: Optional[torch.Size] = None
    ):
        ctx.save_for_backward(input, repeats)
        ctx.dim = dim
        assert not repeats.requires_grad
        return _repeat_interleave_forward(
            input=input,
            repeats=repeats,
            repeats_cumsum=None,
            dim=dim,
            output_size=output_size
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, repeats = ctx.saved_tensors
        dim = ctx.dim
        return _repeat_interleave_backward(
            grad=grad_output,
            input=input,
            repeats=repeats,
            repeats_cumsum=None,
            dim=dim
        ), None, None, None


def repeat_interleave(
    input: torch.Tensor,
    repeats: torch.Tensor,
    dim: Optional[int] = None,
    output_size: Optional[torch.Size] = None
):
    assert input.dim() <= 3, input.dim()
    assert dim in (0, 1), dim
    assert input.is_cuda, input.device

    padded_size = input.size() + (1,) * (3 - input.dim())
    output: torch.Tensor = RepeatInterleave.apply(input.view(*padded_size), repeats, dim, output_size)

    return output.view(*output.shape[:input.dim()])


def pnp_repeat(x, repeats, output_size=None, triton_pnp_repeat=True):
    if x.size(0) == 0:
        return x

    if triton_pnp_repeat:
        return repeat_interleave(x, repeats=repeats, dim=0, output_size=output_size)
    else:
        return torch.cat([xx.repeat(repeat, *([1] * (x.dim() - 1))) for xx, repeat in zip(x, repeats)])
