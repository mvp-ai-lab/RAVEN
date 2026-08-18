"""Convert a safetensors/torch weight file into a sharded DCP checkpoint.

Why this exists, in one measurement: loading a 124 GB fp32 safetensors into a
64-rank FSDP2 model takes 753 s, while a plain sequential read of the same file
takes 31 s and a 371 GB DCP checkpoint loads in 119-179 s on the same cluster.
The file is not the problem and neither is the filesystem -- the problem is that
64 ranks fault their way through one file via mmap, each needing a slice of every
tensor. DCP fixes that by storing the slices: each writer emits its own data
file, and on load each rank streams the chunk it owns.

That only works if there are many writers, so this runs distributed. It is CPU
and gloo only -- no model is instantiated and no GPU is touched, so it belongs on
a cpu partition rather than holding GPUs idle.

Each rank mmaps the same source (virtual, so no RAM is spent until touched),
builds ``Shard(0)`` DTensors over a CPU mesh, and writes its own shard. The
``src_data_rank=None`` is what keeps that honest: it tells DTensor every rank
already holds the correct data, so no rank broadcasts the tensor it just mapped.
Per-rank RAM is therefore about source_size / world_size.

The shard count is the process count, and it need not match the training world
size: DCP reshards on load. Matching it is still worth doing when known, because
an aligned load has every rank read exactly one contiguous chunk, while a
resharded one reassembles each rank's range from several chunks.

    torchrun --nproc_per_node=16 projects/minimax_h3/tools/convert_to_dcp.py \
        --src  /path/to/weights.safetensors \
        --out  /path/to/weights.dcp

The RELEASED MiniMax-H3 DiT needs an explicit layout conversion:

    torchrun --nproc_per_node=16 projects/minimax_h3/tools/convert_to_dcp.py \
        --src  /path/to/FL2VA/transformer/model.safetensors.index.json \
        --out  /path/to/weights.dcp \
        --minimax-h3-dit

Normal released-weight loading goes through ``MiniMaxH3WeightLoader``, which
prefixes the bare DiT keys with ``dit.`` and rewrites per-head ``[q, k, v]`` rows
into the model's ``[q_all, k_all, v_all]`` layout. ``ShardedDCPWeights`` bypasses
that loader, so the flag performs those transforms before writing DCP.

Do NOT use ``--minimax-h3-dit`` for a source already written in model layout:
a training run's ``checkpoints/NNNNNNN`` DCP (which is already rejected as a
source here), or an export made from a loaded model. The qkv permutation is not
idempotent.

A sharded source is given by its index: ``--src .../model.safetensors.index.json``.
A DCP source is rejected -- it is already the target format.

The result is consumed by ``common.plugin.dcp_weights.ShardedDCPWeights`` as a
placement plugin, with no ``weight`` node on the model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def log(message: str) -> None:
    if dist.get_rank() == 0:
        print(f"[convert_to_dcp] {message}", flush=True)


def load_source(src: Path) -> dict[str, torch.Tensor]:
    """Read the source into a state dict, memory-mapped wherever the format allows.

    Every rank calls this on the same path. safetensors is inherently mmap-backed
    and ``torch.load(mmap=True)`` is too, so the returned tensors cost address
    space rather than memory until something reads them -- which is what keeps
    per-rank RAM at source_size / world_size once only the local shards are
    touched.
    """
    if src.is_dir():
        raise SystemExit(f"{src} looks like a DCP checkpoint directory; it is already the target format")

    if src.name.endswith(".safetensors.index.json"):
        from safetensors.torch import load_file

        index = json.loads(src.read_text())
        tensors: dict[str, torch.Tensor] = {}
        for shard_file in dict.fromkeys(index["weight_map"].values()):
            tensors.update(load_file(str(src.parent / shard_file), device="cpu"))
        return tensors

    if src.suffix in (".safetensors", ".safetensor"):
        from safetensors.torch import load_file

        return load_file(str(src), device="cpu")

    return torch.load(src, map_location="cpu", mmap=True)


def reorder_minimax_h3_released_qkv(
    weight: torch.Tensor,
    *,
    num_attention_heads: int,
    attention_head_dim: int,
) -> torch.Tensor:
    """Convert released per-head QKV rows to the model's fused Q/K/V blocks.

    Source rows are
    ``[num_attention_heads, 3, attention_head_dim, ...]`` flattened as
    ``[q0, k0, v0, q1, k1, v1, ...]``. Target rows are
    ``[3, num_attention_heads, attention_head_dim, ...]`` flattened as
    ``[all q | all k | all v]``.
    """
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            "MiniMax-H3 qkv weight has incompatible output dimension: "
            f"got {tuple(weight.shape)}, expected first dimension {expected_rows}"
        )

    rest_shape = weight.shape[1:]
    # Derived from _copy_grouped_qkv_tp_shard at TP world size 1:
    #   grouped = loaded_weight.view(num_query_groups, 3, head_dim, *rest_shape)
    #   target = param.data.view(3, local_groups, head_dim, *rest_shape)
    grouped = weight.reshape(
        num_attention_heads, 3, attention_head_dim, *rest_shape
    )
    target_order = (1, 0, 2, *range(3, grouped.ndim))
    return grouped.permute(target_order).reshape(expected_rows, *rest_shape)


def self_check_minimax_h3_qkv_reorder() -> None:
    num_attention_heads = 3
    attention_head_dim = 2
    source = torch.tensor(
        [
            (head, role, row)
            for head in range(num_attention_heads)
            for role in range(3)
            for row in range(attention_head_dim)
        ]
    )
    expected = torch.tensor(
        [
            (head, role, row)
            for role in range(3)
            for head in range(num_attention_heads)
            for row in range(attention_head_dim)
        ]
    )
    actual = reorder_minimax_h3_released_qkv(
        source,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
    )
    assert torch.equal(actual, expected), (
        "MiniMax-H3 qkv regrouping self-check failed: expected exact "
        "[all q | all k | all v] row ordering"
    )


def load_minimax_h3_arch(path: Path) -> tuple[int, int]:
    import yaml

    arch = yaml.safe_load(path.read_text())
    return int(arch["num_attention_heads"]), int(arch["attention_head_dim"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source .safetensors / .safetensors.index.json / torch file")
    parser.add_argument("--out", required=True, help="Output DCP checkpoint directory")
    parser.add_argument("--key", default=None, help="Optional prefix to store keys under, e.g. models.backbone")
    parser.add_argument(
        "--minimax-h3-dit",
        action="store_true",
        help="Convert the RELEASED MiniMax-H3 DiT from checkpoint layout to model layout",
    )
    parser.add_argument(
        "--arch",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "dit.yaml",
        help="MiniMax-H3 DiT architecture YAML (used only with --minimax-h3-dit)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Cast every tensor to this torch dtype before writing (e.g. float32). "
            "Default: keep the source dtype. Needed when the target model is TRAINABLE "
            "and holds an fp32 master: ShardedDCPWeights fills through Tensor.copy_, so a "
            "bf16 archive would leave that master carrying only bf16 precision, and the "
            "plugin asserts rather than let that happen silently."
        ),
    )
    args = parser.parse_args()
    cast_dtype = getattr(torch, args.dtype) if args.dtype else None

    dit_arch = None
    if args.minimax_h3_dit:
        self_check_minimax_h3_qkv_reorder()
        dit_arch = load_minimax_h3_arch(args.arch)

    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    src, out = Path(args.src), Path(args.out)

    start = time.perf_counter()
    tensors = load_source(src)
    log(f"source {src} mapped: {len(tensors)} tensors, world_size={world_size}")

    mesh = init_device_mesh("cpu", (world_size,))
    sharded: dict[str, torch.Tensor] = {}
    for fqn, tensor in tensors.items():
        if dit_arch is not None:
            num_attention_heads, attention_head_dim = dit_arch
            if fqn.endswith(".qkv_proj.weight"):
                tensor = reorder_minimax_h3_released_qkv(
                    tensor,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
            fqn = f"dit.{fqn}"

        # A 0-dim tensor has no dimension to shard; replicating it stores one
        # copy and still reads back correctly at any world size.
        placement = Replicate() if tensor.ndim == 0 else Shard(0)
        key = f"{args.key}.{fqn}" if args.key else fqn
        dtensor = distribute_tensor(tensor, mesh, [placement], src_data_rank=None)
        # Cast AFTER sharding, so each rank upcasts only its own slice. Casting
        # the source first would materialize a full-size copy on all 16 ranks at
        # once, for a tensor they then throw away all but 1/16 of.
        sharded[key] = dtensor.to(cast_dtype) if cast_dtype is not None else dtensor

    if dist.get_rank() == 0:
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dcp.save(sharded, checkpoint_id=str(out))
    dist.barrier()

    if dist.get_rank() == 0:
        written = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
        log(f"wrote {written / 2**30:.1f} GiB across {len(list(out.glob('*.distcp')))} files "
            f"to {out} in {time.perf_counter() - start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
