#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

sudo apt update
sudo apt install -y python3-venv python3-dev build-essential ffmpeg libatlas-base-dev libhdf5-dev libjpeg-dev zlib1g-dev

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
if ! python -m pip install -r requirements-pi.txt; then
  echo "WARNING: Full requirements installation failed."
  echo "Trying base dependencies without TensorFlow..."
  python -m pip install fastapi uvicorn python-multipart numpy==1.23.5 h5py opencv-python-headless scikit-learn pandas scipy grpcio
  echo "TensorFlow may need a Pi/OS-specific wheel."
  echo "Install TensorFlow manually, then run the dashboard script."
fi

chmod +x run_dashboard_pi.sh

echo "Setup complete."
echo "Next: copy .env.pi.example to .env.pi and adjust paths."
