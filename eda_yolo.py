"""
EDA cho dataset YOLO: tính toán thống kê và vẽ biểu đồ/heatmap

- Đọc nhãn từ `dataset_yolo/labels/{train,val}`
- Tính: width, height, area (đều normalized), aspect_ratio (w/h), center_x, center_y
- Thống kê: phân bố kích thước, tỉ lệ, số object/ảnh, phân bố vị trí (heatmap)
- Lưu hình vào `reports/eda`
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml


def read_yolo_labels(split_dir: Path) -> List[Tuple[str, List[List[float]]]]:
    """Đọc toàn bộ file .txt theo chuẩn YOLO trong thư mục split (train/val).

    Returns danh sách (image_stem, [[cls, cx, cy, w, h], ...]).
    """
    txt_files = sorted((split_dir).glob('*.txt'))
    results: List[Tuple[str, List[List[float]]]] = []
    for txt_path in tqdm(txt_files, desc=f"Đọc nhãn từ {split_dir.name}"):
        boxes: List[List[float]] = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5 and len(parts) != 6:
                        # Hỗ trợ cả format có/không confidence; ưu tiên 5 trường (cls cx cy w h)
                        continue
                    # Lấy 5 trường đầu
                    vals = parts[:5]
                    cls, cx, cy, w, h = map(float, vals)
                    boxes.append([cls, cx, cy, w, h])
        except Exception:
            # Bỏ qua file lỗi
            pass
        results.append((txt_path.stem, boxes))
    return results


def build_dataframe(
    labels_root: Path,
    images_root: Path,
    splits: List[str] = None,
) -> pd.DataFrame:
    """Tạo DataFrame chứa thông tin từ nhãn YOLO.

    Columns: [split, image_id, class_id, cx, cy, w, h]
    Tất cả cx,cy,w,h đều normalized về [0,1].
    """
    if splits is None:
        splits = ["train", "val"]

    rows: List[Dict] = []
    for split in splits:
        split_labels = labels_root / split
        if not split_labels.exists():
            continue
        items = read_yolo_labels(split_labels)
        for image_stem, boxes in items:
            for box in boxes:
                class_id, cx, cy, w, h = box
                rows.append(
                    {
                        "split": split,
                        "image_id": image_stem,
                        "class_id": int(class_id),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(w),
                        "h": float(h),
                    }
                )

    df = pd.DataFrame(rows)
    return df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_size_distributions(df: pd.DataFrame, outdir: Path) -> None:
    """Phân bố width, height, area (normalized)."""
    if df.empty:
        return
    ensure_outdir(outdir)

    df = df.copy()
    df["area"] = df["w"] * df["h"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df["w"], bins=50, kde=True, stat="density")
    plt.title("Phân bố bbox width (normalized)")
    plt.xlabel("width")

    plt.subplot(1, 3, 2)
    sns.histplot(df["h"], bins=50, kde=True, stat="density")
    plt.title("Phân bố bbox height (normalized)")
    plt.xlabel("height")

    plt.subplot(1, 3, 3)
    sns.histplot(df["area"], bins=50, kde=True, stat="density")
    plt.title("Phân bố bbox area (w*h)")
    plt.xlabel("area")

    plt.tight_layout()
    plt.savefig(outdir / "size_distributions.png", dpi=200)
    plt.close()


def plot_aspect_ratio(df: pd.DataFrame, outdir: Path) -> None:
    """Phân bố tỉ lệ w/h và log(tỉ lệ)."""
    if df.empty:
        return
    ensure_outdir(outdir)

    df = df.copy()
    # Tránh chia cho 0
    df = df[df["h"] > 0]
    df["aspect_ratio"] = df["w"] / df["h"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["aspect_ratio"], bins=60, kde=True)
    plt.title("Phân bố aspect ratio (w/h)")
    plt.xlabel("w/h")

    plt.subplot(1, 2, 2)
    sns.histplot(np.log(df["aspect_ratio"]), bins=60, kde=True)
    plt.title("Phân bố log(w/h)")
    plt.xlabel("log(w/h)")

    plt.tight_layout()
    plt.savefig(outdir / "aspect_ratio.png", dpi=200)
    plt.close()


def plot_position_heatmap(df: pd.DataFrame, outdir: Path, bins: int = 50) -> None:
    """Heatmap mật độ vị trí center (cx, cy)."""
    if df.empty:
        return
    ensure_outdir(outdir)

    heat, xedges, yedges = np.histogram2d(df["cx"], df["cy"], bins=bins, range=[[0, 1], [0, 1]])
    heat = heat.T  # để (y, x)

    plt.figure(figsize=(6, 5))
    sns.heatmap(heat, cmap="magma", cbar=True)
    plt.title("Heatmap vị trí tâm bbox (cx, cy)")
    plt.xlabel("cx bins")
    plt.ylabel("cy bins")
    plt.tight_layout()
    plt.savefig(outdir / "position_heatmap.png", dpi=200)
    plt.close()


def plot_objects_per_image(df: pd.DataFrame, outdir: Path) -> None:
    """Phân bố số object trên mỗi ảnh."""
    if df.empty:
        return
    ensure_outdir(outdir)

    counts = df.groupby(["split", "image_id"]).size().reset_index(name="num_objects")

    plt.figure(figsize=(12, 4))
    for i, split in enumerate(sorted(counts["split"].unique())):
        plt.subplot(1, len(counts["split"].unique()), i + 1)
        sns.histplot(counts[counts["split"] == split]["num_objects"], bins=30, discrete=True)
        plt.title(f"Số object/ảnh ({split})")
        plt.xlabel("#objects")
    plt.tight_layout()
    plt.savefig(outdir / "objects_per_image.png", dpi=200)
    plt.close()


def plot_width_height_scatter(df: pd.DataFrame, outdir: Path, sample: int = 5000) -> None:
    """Scatter (w, h) để xem xu hướng kích thước.
    Lấy mẫu để vẽ nhanh nếu dữ liệu lớn.
    """
    if df.empty:
        return
    ensure_outdir(outdir)

    df_sample = df.sample(n=min(sample, len(df)), random_state=42) if len(df) > sample else df
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df_sample, x="w", y="h", s=10, alpha=0.3)
    plt.title("Scatter width vs height (normalized)")
    plt.xlabel("w")
    plt.ylabel("h")
    plt.tight_layout()
    plt.savefig(outdir / "width_height_scatter.png", dpi=200)
    plt.close()


def summarize_print(df: pd.DataFrame) -> None:
    if df.empty:
        print("Không tìm thấy nhãn nào.")
        return
    df_stats = {
        "num_boxes": len(df),
        "avg_w": df["w"].mean(),
        "avg_h": df["h"].mean(),
        "avg_area": (df["w"] * df["h"]).mean(),
        "avg_aspect_ratio": (df["w"] / (df["h"] + 1e-9)).mean(),
    }
    print("=" * 60)
    print("📊 Dataset EDA (YOLO normalized)")
    print("=" * 60)
    for k, v in df_stats.items():
        print(f"{k}: {v:.6f}")

    per_split = df.groupby("split").size().to_dict()
    print("Boxes per split:", per_split)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EDA cho dataset YOLO")
    parser.add_argument("--data", default="dataset_yolo/data.yaml", help="Đường dẫn data.yaml")
    parser.add_argument("--outdir", default="reports/eda", help="Thư mục lưu hình")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="Các split để phân tích")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    assert data_yaml.exists(), f"Không tìm thấy {data_yaml}"

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    root = data_yaml.parent
    labels_root = root / "labels"
    images_root = root / "images"

    df = build_dataframe(labels_root=labels_root, images_root=images_root, splits=args.splits)

    summarize_print(df)

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Plots
    plot_size_distributions(df, outdir)
    plot_aspect_ratio(df, outdir)
    plot_position_heatmap(df, outdir)
    plot_objects_per_image(df, outdir)
    plot_width_height_scatter(df, outdir)

    print(f"\nĐã lưu hình EDA vào: {outdir}")


if __name__ == "__main__":
    main()




