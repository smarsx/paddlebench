#!/usr/bin/env bash
set -euo pipefail

# Use sudo only if not running as root
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

# --- Install prerequisites ---
$SUDO apt update
$SUDO apt install -y vim curl python3-pip git openssh-client ca-certificates ufw

# Ensure ~/.local/bin is on PATH for *this* script run
export PATH="$HOME/.local/bin:$PATH"

# Ensure virtualenv installed (may land in ~/.local/bin)
if ! command -v virtualenv >/dev/null 2>&1; then
  python3 -m pip install --upgrade pip --user
  python3 -m pip install virtualenv --user
fi

# if user is ubuntu instead of root try to add virtualenv to path
if [ "$(id -un)" = "ubuntu" ]; then
  UBASH="/home/ubuntu/.bashrc"
  # Append once (idempotent)
  if ! grep -q 'export PATH="\$HOME/.local/bin:\$PATH"' "$UBASH"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$UBASH"
  fi
fi

# --- Clone repo ---
if [ ! -d "paddlebench" ]; then
  git clone https://github.com/smarsx/paddlebench
fi
cd paddlebench

# --- Python virtual environment ---
virtualenv venv
. venv/bin/activate

# --- 4. Detect CUDA version (from driver) and require 12.9+ for RTX 5090 ---
detect_cuda_ver() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | awk '{print $3}')"
  fi
  echo "${ver}"
}

vstr="$(detect_cuda_ver || true)"
if [ -z "${vstr}" ]; then
  echo "Could not detect CUDA version (nvidia-smi missing). Ensure NVIDIA driver is installed and try again."
  exit 1
fi
echo "Detected driver-reported CUDA version: ${vstr}"

# Normalize like "12.9" -> "129"
vint="$(echo "$vstr" | sed -E 's/[^0-9.]//g; s/^([0-9]+)\.([0-9]+)$/\1\2/')"
if [ "$vint" -lt 129 ]; then
  echo "ERROR: RTX 5090 (Blackwell) needs a driver compatible with CUDA 12.9+."
  echo "Your driver reports CUDA ${vstr}. Please upgrade NVIDIA driver, then re-run."
  exit 1
fi

# --- Install PaddlePaddle GPU (nightly cu129) and PaddleOCR ---
python -m pip install --upgrade pip setuptools wheel

# Force nightly cu129 wheel for PaddlePaddle (works with Blackwell)
python -m pip install --pre paddlepaddle-gpu \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/

pip install -r requirements.txt

# --- Quick smoke test ---
python - <<'PY'
import paddle
print("Paddle version:", paddle.__version__)
paddle.utils.run_check()
print("Paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=True, lang='en')
print("PaddleOCR ready")
PY
