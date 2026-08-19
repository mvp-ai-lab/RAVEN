#!/bin/bash
set -exo pipefail

eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-raven}"
conda activate "$CONDA_ENV"

PREFIX="${PREFIX:-$HOME}/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PREFIX/uv}"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export MAX_JOBS=${MAX_JOBS:-32}
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="90"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="90"
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-TRUE}"

[ ! -d "venv/" ] && uv venv "venv/" --python="$(command -v python)"
uv pip sync tools/requirements.lock \
    --python "venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --link-mode=hardlink
source "venv/bin/activate"

if ! ls assets/flash_attn-2*.whl >/dev/null 2>&1; then
    pip wheel -v git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
fi
pip install assets/flash_attn-2*.whl

if ! ls assets/flash_attn_3*.whl >/dev/null 2>&1; then
    pip wheel -v git+https://github.com/Dao-AILab/flash-attention.git@e2743ab5b3803bb672b16437ba98a3b1d4576c50#subdirectory=hopper --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
fi
pip install assets/flash_attn_3*.whl

if ! ls assets/magi_attention-*.whl >/dev/null 2>&1; then
    export MAGI_ATTENTION_PREBUILD_FFA_JOBS=$MAX_JOBS
    pip wheel -v git+https://github.com/SandAI-org/MagiAttention.git@e08bea8a051031978dbfcd069e2d876b36559bb9 --no-build-isolation --no-deps --wheel-dir=./assets --no-cache-dir
fi
pip install assets/magi_attention-*.whl
