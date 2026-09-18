#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_ENV="$PROJECT_ROOT/.venv-training"
SD_SCRIPTS_DIR="$PROJECT_ROOT/.tools/sd-scripts"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ ! -f "$SD_SCRIPTS_DIR/sdxl_train_network.py" ]]; then
  echo "Pinned sd-scripts checkout is missing; run scripts/bootstrap_training_backend.py first." >&2
  exit 1
fi

if [[ ! -x "$TRAINING_ENV/bin/python" ]]; then
  conda create --yes --prefix "$TRAINING_ENV" python=3.10 pip
fi

"$TRAINING_ENV/bin/python" -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$TRAINING_ENV/bin/python" -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  -r "$PROJECT_ROOT/requirements-training.txt"
"$TRAINING_ENV/bin/python" -m pip install \
  --index-url "$PYPI_INDEX_URL" \
  --no-build-isolation --no-deps --editable "$SD_SCRIPTS_DIR"

"$TRAINING_ENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
print("GPU", torch.cuda.get_device_name(0))
print("BF16 supported", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("Selected GPU does not support BF16")
PY
