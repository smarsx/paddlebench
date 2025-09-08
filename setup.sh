#!/usr/bin/env bash
set -euo pipefail

# Use sudo only if not running as root
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi

# --- 1. Install prerequisites ---
$SUDO apt update
$SUDO apt install -y vim curl python3-pip git openssh-client ca-certificates

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

# --- 5. Python virtual environment ---
# Use the virtualenv we ensured above
virtualenv venv
. venv/bin/activate

# --- Detect CUDA version & bucket down ---
detect_cuda_ver() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | awk '{print $3}')"
  fi
  echo "${ver}"
}

vstr="$(detect_cuda_ver || true)"
if [ -z "${vstr}" ]; then
  echo "Could not detect CUDA version (nvcc/nvidia-smi missing)."
  echo "You can set CUDA_VERSION env var (e.g., 12.6) and re-run."
  exit 1
fi
echo "Detected CUDA version: ${vstr}"

# Normalize like "12.9" -> "129", "11.8" -> "118"
vint="$(echo "$vstr" | sed -E 's/[^0-9.]//g; s/^([0-9]+)\.([0-9]+)$/\1\2/')"

# Buckets: <=11.8 -> cu118
# 11.8 < v < 12.6 -> cu118
# 12.6 <= v < 12.9 -> cu126
# >=12.9 -> cu129
paddle_idx=""
if [ "$vint" -lt 118 ]; then
  paddle_idx="cu118"
elif [ "$vint" -ge 118 ] && [ "$vint" -lt 126 ]; then
  paddle_idx="cu118"
elif [ "$vint" -ge 126 ] && [ "$vint" -lt 129 ]; then
  paddle_idx="cu126"
else
  paddle_idx="cu129"
fi

echo "Using Paddle index bucket: ${paddle_idx}"

# --- Install Paddle & deps ---
python -m pip install --upgrade pip setuptools wheel
python -m pip install "paddlepaddle-gpu==3.1.0" -i "https://www.paddlepaddle.org.cn/packages/stable/${paddle_idx}/"
pip install -r requirements.txt
