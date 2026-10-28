#!/bin/zsh
# REVIEWABLE PROPOSAL ONLY. Not executed; requires explicit user approval.
set -eu
encoder_root='/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/encoder-readiness-20261021'
encoder_runtime="$encoder_root/runtime"
if [[ -e "$encoder_runtime" ]]; then
  print -u2 'Existing runtime is preserved; inspect it before a retry.'
  exit 1
fi
mkdir "$encoder_runtime"
export UV_CACHE_DIR="$encoder_runtime/uv-cache"
export TORCH_HOME="$encoder_runtime/torch-cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
/opt/homebrew/bin/uv --no-config venv --no-project --no-python-downloads --python /Users/lkc/miniforge3/bin/python --link-mode copy "$encoder_runtime/.venv"
/opt/homebrew/bin/uv --no-config pip install --python "$encoder_runtime/.venv/bin/python" --no-python-downloads --no-deps --require-hashes --link-mode copy -r "$encoder_root/proposed-requirements-lock.txt"
"$encoder_runtime/.venv/bin/python" -B "$encoder_root/fetch-weight-after-approval.py" "$encoder_runtime"
# Calibration and mechanism evaluation are separate, gated steps.
