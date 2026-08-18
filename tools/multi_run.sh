#!/bin/bash
set -exo pipefail

eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-base}"
conda activate "$CONDA_ENV"
source "venv/bin/activate"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

N=${N:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}}  # nproc per node
D=${D:-0}  # debug mode
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=31
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:-${SLURM_NNODES:-1}}
NPROC_PER_NODE=$N
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
LOCAL_ADDR=${LOCAL_ADDR:-$(python -c "import socket; print(socket.gethostbyname('$(hostname)'))")}
if [ -z "${MASTER_ADDR}" ]; then
    if [ -n "${SLURM_JOB_NODELIST}" ]; then
        MASTER_ADDR=$(python -c "import hostlist; print(hostlist.expand_hostlist('$SLURM_JOB_NODELIST')[0])")
    else
        MASTER_ADDR="localhost"
    fi
fi
if [ -z "${MASTER_PORT}" ]; then
    if [ -n "${SLURM_JOB_ID}" ]; then
        MASTER_PORT=$(expr 10000 + $SLURM_JOB_ID % 20000)
    else
        MASTER_PORT=7890
    fi
fi

RDZV_BACKEND=static
if [ $NNODES -gt 1 ]; then
    RDZV_BACKEND=c10d
fi

if [ $D -eq 0 ]; then
    torchrun \
        --nnodes $NNODES \
        --nproc_per_node $NPROC_PER_NODE \
        --node_rank $NODE_RANK \
        --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv-backend $RDZV_BACKEND \
        --local_addr=$LOCAL_ADDR \
        -m common.launch --config $@
else
    DEBUGPY_PORT=${DEBUGPY_PORT:-5678}
    export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=30
    export CUDA_LAUNCH_BLOCKING=1
    python -m debugpy --listen 0.0.0.0:$DEBUGPY_PORT --wait-for-client $(which torchrun) \
        --nnodes 1 \
        --nproc_per_node $D \
        --master-port $MASTER_PORT \
        --monitor-interval 0.1 \
        -m common.launch --config $@
fi
