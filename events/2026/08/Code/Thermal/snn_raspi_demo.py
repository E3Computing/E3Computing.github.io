#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi + MLX90640 + SNN Real-time Human Detection Demo
============================================================

目的:
  - 測定データは保存しない
  - 推論結果も保存しない
  - SNNでリアルタイム人検知だけを行う
  - GUIで「魅せる」ためのリアルタイム可視化を行う

想定ハードウェア:
  - Raspberry Pi 5 / Raspberry Pi OS 64-bit with GUI
  - Pimoroni / Adafruit MLX90640 thermal camera, 32x24

推奨実行例:
  python3 raspi_thermal_snn_demo.py \
    --model-path /home/userhome/m5291066/thermal_human_recognition/models/deployed/snn_model.pt \
    --threshold 0.51

TorchScriptモデルを使う場合:
  python3 raspi_thermal_snn_demo.py \
    --model-path ./snn_demo_model.pt \
    --model-type torchscript \
    --threshold 0.51

終了:
  GUIウィンドウ上で q キー / ESC キー
"""

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Optional hardware imports
# ============================================================
try:
    import board
    import busio
    import adafruit_mlx90640
    HAS_MLX90640 = True
except Exception:
    HAS_MLX90640 = False


# ============================================================
# Simple IF-based SNN demo model
# ============================================================
# 注意:
#   ここは「state_dictだけ保存したSNN」を読み込むための予備クラスです。
#   研究で使ったSNNの層名・構造が違う場合、state_dictは一致しません。
#   その場合は、学習済みモデルをTorchScript化して --model-type torchscript で使うのが一番安全です。
#
# 入力:
#   x: [B, 1, 24, 32]
# 出力:
#   logit: [B, 1]

class IFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float):
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DemoSNN(nn.Module):
    def __init__(self, timesteps: int = 10):
        super().__init__()
        self.timesteps = timesteps

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 3 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

        self.spike_threshold = 1.0

    def _if_layer(self, current: torch.Tensor, membrane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        membrane = membrane + current
        spike = IFSpike.apply(membrane, self.spike_threshold)
        membrane = membrane * (1.0 - spike)
        return spike, membrane

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rate-like temporal replication
        out_sum = 0.0

        mem1 = mem2 = mem3 = mem_fc1 = None

        for _ in range(self.timesteps):
            z1 = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            if mem1 is None:
                mem1 = torch.zeros_like(z1)
            s1, mem1 = self._if_layer(z1, mem1)

            z2 = self.pool2(torch.relu(self.bn2(self.conv2(s1))))
            if mem2 is None:
                mem2 = torch.zeros_like(z2)
            s2, mem2 = self._if_layer(z2, mem2)

            z3 = self.pool3(torch.relu(self.bn3(self.conv3(s2))))
            if mem3 is None:
                mem3 = torch.zeros_like(z3)
            s3, mem3 = self._if_layer(z3, mem3)

            flat = s3.flatten(1)
            z4 = torch.relu(self.fc1(flat))
            if mem_fc1 is None:
                mem_fc1 = torch.zeros_like(z4)
            s4, mem_fc1 = self._if_layer(z4, mem_fc1)

            out_sum = out_sum + self.fc2(s4)

        return out_sum / float(self.timesteps)


# ============================================================
# Sensor / dummy frame source
# ============================================================

class MLX90640Reader:
    def __init__(self, refresh_rate: int = 8):
        if not HAS_MLX90640:
            raise RuntimeError(
                "MLX90640 library not found. Install with: pip install adafruit-circuitpython-mlx90640"
            )

        i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
        self.mlx = adafruit_mlx90640.MLX90640(i2c)

        if refresh_rate == 2:
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        elif refresh_rate == 4:
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        elif refresh_rate == 16:
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        elif refresh_rate == 32:
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ
        else:
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

        self.frame = [0.0] * 768

    def read(self) -> np.ndarray:
        # MLX90640はたまにValueErrorを出すので、短くリトライする
        for _ in range(5):
            try:
                self.mlx.getFrame(self.frame)
                arr = np.array(self.frame, dtype=np.float32).reshape(24, 32)
                return arr
            except ValueError:
                time.sleep(0.01)
        raise RuntimeError("Failed to read MLX90640 frame")


class DummyThermalReader:
    """実機なしでGUI表示だけ確認するためのダミー入力。"""

    def __init__(self):
        self.t = 0

    def read(self) -> np.ndarray:
        h, w = 24, 32
        yy, xx = np.mgrid[0:h, 0:w]
        base = 24.0 + np.random.normal(0, 0.15, size=(h, w)).astype(np.float32)

        cx = 16 + 7 * math.sin(self.t * 0.08)
        cy = 12 + 4 * math.cos(self.t * 0.05)
        blob = 8.0 * np.exp(-(((xx - cx) ** 2) / 18.0 + ((yy - cy) ** 2) / 28.0))

        self.t += 1
        time.sleep(0.06)
        return (base + blob).astype(np.float32)


# ============================================================
# Preprocess / model loading / inference
# ============================================================

def preprocess_frame(frame_celsius: np.ndarray) -> torch.Tensor:
    """
    24x32 thermal frame -> tensor [1, 1, 24, 32]

    デモ用途ではフレーム内min-max正規化にして、表示と推論の見た目を安定させる。
    研究時の realtime_inference_normalized.py と正規化方法が違う場合は、ここを合わせる。
    """
    x = frame_celsius.astype(np.float32)
    x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmax(x), neginf=np.nanmin(x))

    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-6:
        x = np.zeros_like(x, dtype=np.float32)
    else:
        x = (x - mn) / (mx - mn)

    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,24,32]
    return tensor.float()


def load_model(model_path: Path, model_type: str, device: torch.device, timesteps: int) -> nn.Module:
    if model_type == "torchscript":
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model

    model = DemoSNN(timesteps=timesteps).to(device)
    checkpoint = torch.load(str(model_path), map_location=device)

    # よくある保存形式に対応
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError(
            "This file does not look like a state_dict checkpoint. "
            "Use --model-type torchscript if this is a scripted/traced model."
        )

    # DataParallel由来の module. を除去
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")

        if new_k.startswith("snn."):
            new_k = new_k[len("snn."):]

        if ".lif" in new_k or new_k.startswith("lif"):
            continue

        clean_state_dict[new_k] = v
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_probability(model: nn.Module, x: torch.Tensor, device: torch.device) -> float:
    x = x.to(device)
    y = model(x)

    # 出力がtuple/listの場合にも一応対応
    if isinstance(y, (tuple, list)):
        y = y[0]

    y = y.reshape(-1)[0]
    prob = torch.sigmoid(y).item()
    return float(prob)


# ============================================================
# GUI rendering
# ============================================================

def to_heatmap(frame_celsius: np.ndarray, out_w: int = 640, out_h: int = 480) -> np.ndarray:
    x = frame_celsius.astype(np.float32)
    x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmax(x), neginf=np.nanmin(x))

    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-6:
        norm = np.zeros_like(x, dtype=np.uint8)
    else:
        norm = ((x - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    big = cv2.resize(norm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)
    return heat


def draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, label: str):
    value = max(0.0, min(1.0, value))
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (230, 230, 230), 2)

    fill_w = int(w * value)
    # OpenCVはBGR。赤寄りほどHuman感が出る。
    color = (40, 220, 80) if value < 0.5 else (40, 160, 255)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)

    text = f"{label}: {value * 100:5.1f}%"
    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def draw_demo_ui(
    frame_celsius: np.ndarray,
    prob: float,
    threshold: float,
    fps: float,
    infer_ms: float,
    temp_min: float,
    temp_max: float,
    history: deque,
) -> np.ndarray:
    heat = to_heatmap(frame_celsius, 640, 480)

    canvas_h, canvas_w = 720, 1100
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 22)

    # 左: thermal heatmap
    canvas[120:600, 40:680] = heat
    cv2.rectangle(canvas, (40, 120), (680, 600), (235, 235, 235), 2)

    # タイトル
    cv2.putText(canvas, "Thermal SNN Human Detection Demo", (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Raspberry Pi + MLX90640 + Spiking Neural Network", (42, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 190), 2, cv2.LINE_AA)

    # 判定
    is_human = prob >= threshold
    status = "HUMAN DETECTED" if is_human else "NO HUMAN"
    status_color = (40, 160, 255) if is_human else (70, 220, 120)

    cv2.rectangle(canvas, (720, 120), (1060, 220), status_color, -1)
    cv2.putText(canvas, status, (745, 183),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 3, cv2.LINE_AA)

    # Probability bar
    draw_bar(canvas, 720, 285, 340, 36, prob, "Human probability")

    # Threshold marker
    marker_x = 720 + int(340 * threshold)
    cv2.line(canvas, (marker_x, 278), (marker_x, 330), (255, 255, 255), 2)
    cv2.putText(canvas, f"threshold {threshold:.2f}", (720, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)

    # Metrics panel
    info_y = 420
    infos = [
        f"FPS           : {fps:5.1f}",
        f"Inference     : {infer_ms:5.1f} ms",
        f"Temp range    : {temp_min:4.1f} - {temp_max:4.1f} C",
        f"Input frame   : 24 x 32",
        f"Mode          : Real-time only",
        f"Save data     : OFF",
        f"Save results  : OFF",
    ]
    for i, txt in enumerate(infos):
        cv2.putText(canvas, txt, (720, info_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (230, 230, 230), 2, cv2.LINE_AA)

    # Probability history
    hx, hy, hw, hh = 720, 640, 340, 55
    cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (45, 45, 50), -1)
    cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (180, 180, 180), 1)
    cv2.putText(canvas, "Probability trend", (hx, hy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    if len(history) >= 2:
        pts = []
        values = list(history)
        for i, v in enumerate(values):
            px = hx + int(i / max(1, len(values) - 1) * hw)
            py = hy + hh - int(v * hh)
            pts.append((px, py))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p1, p2, (255, 255, 255), 2)

    # Bottom instruction
    cv2.putText(canvas, "Press q or ESC to quit", (40, 670),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2, cv2.LINE_AA)

    return canvas


# ============================================================
# Main loop
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi Thermal SNN Real-time Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SNN model .pth or TorchScript .pt")
    parser.add_argument("--model-type", type=str, default="state_dict", choices=["state_dict", "torchscript"])
    parser.add_argument("--threshold", type=float, default=0.51, help="Human detection threshold")
    parser.add_argument("--timesteps", type=int, default=10, help="SNN timesteps for DemoSNN state_dict mode")
    parser.add_argument("--refresh-rate", type=int, default=8, choices=[2, 4, 8, 16, 32])
    parser.add_argument("--dummy", action="store_true", help="Use dummy thermal frames without MLX90640")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Torch CPU threads")
    parser.add_argument("--window-name", type=str, default="Thermal SNN Human Detection Demo")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    print("=" * 70)
    print("Thermal SNN Human Detection Demo")
    print("=" * 70)
    print(f"Model path     : {model_path}")
    print(f"Model type     : {args.model_type}")
    print(f"Threshold      : {args.threshold:.3f}")
    print(f"Device         : {device}")
    print(f"Save data      : OFF")
    print(f"Save results   : OFF")
    print("=" * 70)

    model = load_model(model_path, args.model_type, device, args.timesteps)

    if args.dummy:
        reader = DummyThermalReader()
        print("[INFO] Dummy thermal reader enabled.")
    else:
        reader = MLX90640Reader(refresh_rate=args.refresh_rate)
        print("[INFO] MLX90640 reader enabled.")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, 1100, 720)

    prob_history = deque(maxlen=80)
    fps_history = deque(maxlen=20)

    last_loop_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            frame = reader.read()
            temp_min = float(np.nanmin(frame))
            temp_max = float(np.nanmax(frame))

            x = preprocess_frame(frame)

            infer_start = time.perf_counter()
            prob = predict_probability(model, x, device)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            now = time.perf_counter()
            dt = max(now - last_loop_time, 1e-6)
            last_loop_time = now
            fps_history.append(1.0 / dt)
            fps = float(np.mean(fps_history))

            prob_history.append(prob)

            ui = draw_demo_ui(
                frame_celsius=frame,
                prob=prob,
                threshold=args.threshold,
                fps=fps,
                infer_ms=infer_ms,
                temp_min=temp_min,
                temp_max=temp_max,
                history=prob_history,
            )

            cv2.imshow(args.window_name, ui)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # CPU使用率を少し抑える。センサのrefreshより速く回しても意味が薄い。
            elapsed = time.perf_counter() - loop_start
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] demo finished")


if __name__ == "__main__":
    main()

