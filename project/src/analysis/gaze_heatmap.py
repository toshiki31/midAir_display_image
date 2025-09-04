#!/usr/bin/env python3
"""
視線計測CSVからヒートマップ画像を作成します。

入力CSVの想定カラム:
  - left_eye_x,left_eye_y,right_eye_x,right_eye_y

使い方の要点
  基本:
  入力: --input <CSVパス>
  出力: --output 未指定なら <CSV名>_heatmap.png を自動生成
  キャンバスと平滑化:
    --canvas-width, --canvas-height で出力サイズを指定（左上原点）
    --sigma で平滑化の強さ（奇数推奨、例: 21/31）
  カラーマップ:
    --cmap に turbo, jet, hot など Matplotlib の名前を指定可能

使用例:
  python src/analysis/gaze_heatmap.py \
    --input project/2025-09-04_10:37:38_ChAerial_gaze_data.csv \
    --output project/2025-09-04_10:37:38_ChAerial_gaze_heatmap.png \
    --canvas-width 1920 --canvas-height 1080 --bins 300 --sigma 21

ポイント:
  - 画像座標系は左上を(0,0)として生成します（origin='upper'）。
  - 両眼がある行は平均、片眼のみはその片眼を採用します。
  - 値のスケールが0..1に見える場合はそのまま、そうでなければmin/maxで0..1へ線形正規化します。
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def _infer_normalized(xs: np.ndarray, ys: np.ndarray) -> bool:
    """0..1正規化済みかのヒューリスティック判定。

    - p10>=-0.05, p90<=1.05 かつ 0..1範囲に入る割合が60%以上なら正規化とみなす
    """
    xs_f = xs[np.isfinite(xs)]
    ys_f = ys[np.isfinite(ys)]
    if xs_f.size < 10 or ys_f.size < 10:
        return True  # データが少ない場合は正規化前提
    p10x, p90x = np.percentile(xs_f, [10, 90])
    p10y, p90y = np.percentile(ys_f, [10, 90])
    in_x = np.mean((xs_f >= -0.05) & (xs_f <= 1.05))
    in_y = np.mean((ys_f >= -0.05) & (ys_f <= 1.05))
    return (p10x >= -0.05 and p90x <= 1.05 and p10y >= -0.05 and p90y <= 1.05 and (in_x + in_y) / 2.0 >= 0.6)


def _combine_gaze_points(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """両眼の座標を1点に統合する（平均または片眼のみ）。"""
    lx = df["left_eye_x"].to_numpy(dtype=float)
    ly = df["left_eye_y"].to_numpy(dtype=float)
    rx = df["right_eye_x"].to_numpy(dtype=float)
    ry = df["right_eye_y"].to_numpy(dtype=float)

    # 片眼・両眼の有効値をマスク
    l_valid = np.isfinite(lx) & np.isfinite(ly)
    r_valid = np.isfinite(rx) & np.isfinite(ry)

    # 初期化
    x = np.full_like(lx, np.nan, dtype=float)
    y = np.full_like(ly, np.nan, dtype=float)

    # 両眼ある行は平均
    both = l_valid & r_valid
    x[both] = (lx[both] + rx[both]) / 2.0
    y[both] = (ly[both] + ry[both]) / 2.0

    # 片眼のみ
    only_l = l_valid & ~r_valid
    x[only_l] = lx[only_l]
    y[only_l] = ly[only_l]

    only_r = r_valid & ~l_valid
    x[only_r] = rx[only_r]
    y[only_r] = ry[only_r]

    # NaN除去
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _scale_points(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """(xs,ys)を0..canvas-1へスケールしてピクセル座標に変換。"""
    if _infer_normalized(xs, ys):
        xs_n = np.clip(xs, 0.0, 1.0)
        ys_n = np.clip(ys, 0.0, 1.0)
    else:
        # min/maxで0..1へ線形正規化（外れ値があれば単純クリップ）
        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
        y_min, y_max = np.nanmin(ys), np.nanmax(ys)
        # ゼロ割回避
        if x_max - x_min < 1e-6:
            x_max = x_min + 1e-6
        if y_max - y_min < 1e-6:
            y_max = y_min + 1e-6
        xs_n = (xs - x_min) / (x_max - x_min)
        ys_n = (ys - y_min) / (y_max - y_min)
        xs_n = np.clip(xs_n, 0.0, 1.0)
        ys_n = np.clip(ys_n, 0.0, 1.0)

    # ピクセル座標へ（左上(0,0)を原点にするのでyは下が正）
    xi = (xs_n * (width - 1)).astype(int)
    yi = (ys_n * (height - 1)).astype(int)
    return xi, yi


def build_heatmap(xi: np.ndarray, yi: np.ndarray, width: int, height: int, sigma: int = 21) -> np.ndarray:
    """散布からヒストグラムを作り、ガウシアン平滑化したヒートマップを返す。

    sigma: ガウシアンのカーネルサイズ（奇数ピクセル）。大きいほどなめらか。
    """
    heat = np.zeros((height, width), dtype=np.float32)
    # 範囲内のみカウント
    in_bounds = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi_b = xi[in_bounds]
    yi_b = yi[in_bounds]
    # 累積
    np.add.at(heat, (yi_b, xi_b), 1)

    # 平滑化（奇数へ調整）
    if sigma and sigma > 1:
        k = int(sigma)
        if k % 2 == 0:
            k += 1
        heat = cv2.GaussianBlur(heat, (k, k), 0)

    # 0..1正規化
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def render_and_save(heat: np.ndarray, out_path: str, cmap: str = "turbo") -> None:
    plt.figure(figsize=(10, 6), dpi=150)
    plt.imshow(heat, cmap=cmap, origin="upper")  # 左上が(0,0)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="視線CSVからヒートマップを生成")
    parser.add_argument("--input", "-i", required=True, help="入力CSVパス")
    parser.add_argument("--output", "-o", default=None, help="出力画像パス (未指定ならCSV名に_heatmap.pngを付ける)")
    parser.add_argument("--canvas-width", type=int, default=1920, help="キャンバス幅(ピクセル)")
    parser.add_argument("--canvas-height", type=int, default=1080, help="キャンバス高(ピクセル)")
    parser.add_argument("--sigma", type=int, default=21, help="ガウシアン平滑化カーネルサイズ（奇数）")
    parser.add_argument("--cmap", type=str, default="turbo", help="Matplotlibカラーマップ名")
    # 座標系オプション
    parser.add_argument("--flip-x", action="store_true", help="X軸を反転（x=0とx=1を入れ替え）")
    parser.add_argument("--flip-y", action="store_true", help="Y軸を反転（y=0とy=1を入れ替え）")
    parser.add_argument("--swap-xy", action="store_true", help="XとYを入れ替え（列の意味が逆の場合に使用）")
    parser.add_argument(
        "--normalized",
        choices=["auto", "true", "false"],
        default="auto",
        help="入力座標を0..1の正規化として扱うか。auto=自動判定、true=正規化として扱う、false=min/maxで正規化",
    )
    parser.add_argument(
        "--y-origin",
        choices=["top", "bottom"],
        default="top",
        help="y=0の原点位置。top=上（画像座標）、bottom=下（数学座標）。",
    )
    args = parser.parse_args()

    csv_path = args.input
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    # CSV読み込み
    df = pd.read_csv(csv_path)
    required_cols = {"left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSVに必要なカラムがありません。必要: {required_cols}, 実際: {set(df.columns)}")

    xs, ys = _combine_gaze_points(df)
    if xs.size == 0:
        raise ValueError("有効な視線データがありません（全てNaN）")

    # 必要に応じて軸入れ替え
    if args.swap_xy:
        xs, ys = ys.copy(), xs.copy()
    # ピクセル変換前に反転指示を渡すため、内部で正規化→反転→ピクセル化を実施
    def _scale_points_with_flip(xs, ys, width, height, flip_x=False, flip_y=False, normalized_mode="auto", y_origin="top"):
        # _scale_pointsのロジックを利用しつつ、反転を挟む
        is_norm = _infer_normalized(xs, ys) if normalized_mode == "auto" else (normalized_mode == "true")
        if is_norm:
            xs_n = np.clip(xs, 0.0, 1.0)
            ys_n = np.clip(ys, 0.0, 1.0)
        else:
            x_min, x_max = np.nanmin(xs), np.nanmax(xs)
            y_min, y_max = np.nanmin(ys), np.nanmax(ys)
            if x_max - x_min < 1e-6:
                x_max = x_min + 1e-6
            if y_max - y_min < 1e-6:
                y_max = y_min + 1e-6
            xs_n = (xs - x_min) / (x_max - x_min)
            ys_n = (ys - y_min) / (y_max - y_min)
            xs_n = np.clip(xs_n, 0.0, 1.0)
            ys_n = np.clip(ys_n, 0.0, 1.0)

        if flip_x:
            xs_n = 1.0 - xs_n
        if flip_y:
            ys_n = 1.0 - ys_n
        if y_origin == "bottom":
            ys_n = 1.0 - ys_n

        xi = (xs_n * (width - 1)).astype(int)
        yi = (ys_n * (height - 1)).astype(int)
        return xi, yi

    xi, yi = _scale_points_with_flip(
        xs,
        ys,
        args.canvas_width,
        args.canvas_height,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        normalized_mode=args.normalized,
        y_origin=args.y_origin,
    )
    heat = build_heatmap(xi, yi, args.canvas_width, args.canvas_height, sigma=args.sigma)

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(csv_path)
        out_path = f"{base}_heatmap.png"

    render_and_save(heat, out_path, cmap=args.cmap)
    print(f"Saved heatmap -> {out_path}")


if __name__ == "__main__":
    main()
