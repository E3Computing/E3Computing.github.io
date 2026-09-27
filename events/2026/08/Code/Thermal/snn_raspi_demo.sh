#!/usr/bin/env bash
# ============================================================
# Raspberry Pi + MLX90640 + SNN Human Detection Demo Launcher
# ============================================================
#
# 目的:
#   - デモ用Pythonコードを一発起動する
#   - 測定データ保存なし
#   - 推論結果保存なし
#   - GUIでリアルタイム可視化
#
# 使い方:
#   chmod +x run_thermal_snn_demo.sh
#   ./run_thermal_snn_demo.sh
#
# ダミー入力でGUIだけ確認:
#   ./run_thermal_snn_demo.sh --dummy
#
# ============================================================

set -e

# -----------------------------
# Project paths
# -----------------------------
PROJECT_DIR="/home/m5291066/thermal_human_recognition"
SCRIPT_PATH="${PROJECT_DIR}/scripts/demo/snn_raspi_demo.py"

# -----------------------------
# Model settings
# -----------------------------
# state_dictの場合:
MODEL_PATH="${PROJECT_DIR}/models/deployed/snn_raspi.pt"
MODEL_TYPE="state_dict"

# TorchScriptモデルを使う場合は、上の2行を以下のように変更:
# MODEL_PATH="${PROJECT_DIR}/models/snn_demo_model.pt"
# MODEL_TYPE="state_dict"

# SNN threshold
# 以前のRaspberry Pi実験では SNN 推奨 threshold = 0.51
THRESHOLD="0.51"

# SNN timesteps
TIMESTEPS="10"

# MLX90640 refresh rate: 2, 4, 8, 16, 32
REFRESH_RATE="4"

# Raspberry Pi CPU threads
CPU_THREADS="4"

# GUI window title
WINDOW_NAME="Thermal SNN Human Detection Demo"

# -----------------------------
# Optional: virtual environment
# -----------------------------
# venvを使っている場合はここを有効化
# VENV_PATH="${PROJECT_DIR}/venv"
# source "${VENV_PATH}/bin/activate"

# -----------------------------
# Display check
# -----------------------------
if [ -z "${DISPLAY:-}" ]; then
  echo "[WARN] DISPLAY is not set."
  echo "[WARN] GUI環境で実行してください。SSHの場合はVNC/直接画面/ssh -X等が必要です。"
fi

# -----------------------------
# File checks
# -----------------------------
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[ERROR] Demo script not found: $SCRIPT_PATH"
  echo "        SCRIPT_PATHを実際の配置場所に合わせて変更してください。"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[ERROR] Model file not found: $MODEL_PATH"
  echo "        MODEL_PATHを実際の学習済みモデルに合わせて変更してください。"
  exit 1
fi

# -----------------------------
# Launch
# -----------------------------
echo "============================================================"
echo " Thermal SNN Human Detection Demo Launcher"
echo "============================================================"
echo "Project dir   : $PROJECT_DIR"
echo "Script        : $SCRIPT_PATH"
echo "Model         : $MODEL_PATH"
echo "Model type    : $MODEL_TYPE"
echo "Threshold     : $THRESHOLD"
echo "Timesteps     : $TIMESTEPS"
echo "Refresh rate  : $REFRESH_RATE Hz"
echo "CPU threads   : $CPU_THREADS"
echo "Save data     : OFF"
echo "Save results  : OFF"
echo "============================================================"

python3 "$SCRIPT_PATH" \
  --model-path "$MODEL_PATH" \
  --model-type "$MODEL_TYPE" \
  --threshold "$THRESHOLD" \
  --timesteps "$TIMESTEPS" \
  --refresh-rate "$REFRESH_RATE" \
  --cpu-threads "$CPU_THREADS" \
  --window-name "$WINDOW_NAME" \
  "$@"

