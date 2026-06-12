#!/usr/bin/env bash
set -euo pipefail

# Direct environment setup for cloud platforms that already provide a container.
# Run this script from the project root inside the cloud container.

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.2.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.0}"
INSTALL_APT="${INSTALL_APT:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
REQUIRED_PYTHON_MAJOR="${REQUIRED_PYTHON_MAJOR:-3}"
REQUIRED_PYTHON_MINOR="${REQUIRED_PYTHON_MINOR:-10}"
ALLOW_PYTHON_MISMATCH="${ALLOW_PYTHON_MISMATCH:-0}"

"${PYTHON_BIN}" - <<PY
import sys

required = (${REQUIRED_PYTHON_MAJOR}, ${REQUIRED_PYTHON_MINOR})
current = sys.version_info[:2]
allow_mismatch = "${ALLOW_PYTHON_MISMATCH}" == "1"
if current != required and not allow_mismatch:
    raise SystemExit(
        "This project environment is pinned to Python "
        f"{required[0]}.{required[1]}, but {current[0]}.{current[1]} is active. "
        "Please switch the cloud image/template to Python 3.10, or set PYTHON_BIN "
        "to a Python 3.10 executable before running this script. If the cloud "
        "platform only provides Python 3.12 + PyTorch 2.3, rerun with "
        "ALLOW_PYTHON_MISMATCH=1 INSTALL_TORCH=0 and treat the result as "
        "a compatibility smoke-test environment."
    )
if current != required and allow_mismatch:
    print(
        "WARNING: running outside the pinned project environment: "
        f"Python {current[0]}.{current[1]} instead of {required[0]}.{required[1]}. "
        "Use this only for platform-constrained smoke tests."
    )
PY

if [[ "${INSTALL_APT}" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      python3-pip \
      python3-opencv
    rm -rf /var/lib/apt/lists/*
  else
    echo "apt-get not found; skip system dependency installation."
  fi
fi

"${PYTHON_BIN}" -m pip install --upgrade pip

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
else
  echo "Skip PyTorch installation because INSTALL_TORCH=${INSTALL_TORCH}."
fi

"${PYTHON_BIN}" -m pip install -r requirements.txt

"${PYTHON_BIN}" - <<'PY'
import cv2
import torch
import sys

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("PyTorch CUDA is not available. Check the cloud CUDA image and torch wheel.")
PY
