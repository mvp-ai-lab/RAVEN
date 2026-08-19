#!/bin/bash
set -exo pipefail

eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-raven}"
conda activate "$CONDA_ENV"

uv pip compile tools/requirements.txt \
    -o tools/requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
