#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Python virtual environment not found: $VENV_PY"
  echo "Run setup_pi.sh first."
  exit 1
fi

ENV_FILE=""
if [[ -f "$APP_DIR/.env.pi" ]]; then
  ENV_FILE="$APP_DIR/.env.pi"
elif [[ -f "$APP_DIR/env.pi" ]]; then
  ENV_FILE="$APP_DIR/env.pi"
fi

if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  echo "Using config: $ENV_FILE"
else
  echo "No config file found (.env.pi or env.pi). Using defaults."
fi

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s' "$p"
  else
    printf '%s' "$APP_DIR/$p"
  fi
}

pick_default_video_path() {
  local candidates=(
    "$APP_DIR/video.mp4"
    "$APP_DIR/Video Project 5.2.mp4"
    "$APP_DIR/../video.mp4"
    "$APP_DIR/../Video Project 5.2.mp4"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  printf '%s' "$APP_DIR/video.mp4"
}

export MODEL_PATH="${MODEL_PATH:-$APP_DIR/best_model.h5}"
export WORDLIST_PATH="${WORDLIST_PATH:-$APP_DIR/turizm_sozluk.json}"
if [[ -z "${VIDEO_PATH:-}" ]]; then
  export VIDEO_PATH="$(pick_default_video_path)"
else
  export VIDEO_PATH
fi

export MODEL_PATH="$(resolve_path "$MODEL_PATH")"
export WORDLIST_PATH="$(resolve_path "$WORDLIST_PATH")"
export VIDEO_PATH="$(resolve_path "$VIDEO_PATH")"

export PROCESS_EVERY_N_FRAME="${PROCESS_EVERY_N_FRAME:-4}"
export SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-3}"
export DOMINANT_MIN_COUNT="${DOMINANT_MIN_COUNT:-1}"
export MIN_COMMIT_INTERVAL_SEC="${MIN_COMMIT_INTERVAL_SEC:-0.30}"
export COMMIT_CONFIDENCE_THRESHOLD="${COMMIT_CONFIDENCE_THRESHOLD:-0.15}"
export TOP1_TOP2_MARGIN_MIN="${TOP1_TOP2_MARGIN_MIN:-0.02}"
export REPEAT_LABEL_GAP_SEC="${REPEAT_LABEL_GAP_SEC:-0.80}"
export NOTHING_RELEASE_REQUIRED="${NOTHING_RELEASE_REQUIRED:-2}"
export TOP_K="${TOP_K:-5}"
export STREAM_MAX_WIDTH="${STREAM_MAX_WIDTH:-960}"
export STREAM_FRAME_INTERVAL_SEC="${STREAM_FRAME_INTERVAL_SEC:-0.06}"
export JPEG_QUALITY="${JPEG_QUALITY:-75}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$WORDLIST_PATH" ]]; then
  echo "Word list file not found: $WORDLIST_PATH"
  exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video file not found: $VIDEO_PATH"
  echo "Set VIDEO_PATH in .env.pi or env.pi"
  exit 1
fi

echo "Starting dashboard with:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORDLIST_PATH=$WORDLIST_PATH"
echo "  VIDEO_PATH=$VIDEO_PATH"

cd "$APP_DIR"
exec "$VENV_PY" "$APP_DIR/video_dashboard.py"
