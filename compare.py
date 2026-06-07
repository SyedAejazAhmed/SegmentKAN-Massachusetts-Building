#!/usr/bin/env python3
"""
Run the trained UKAN rooftop/building model and save black/white comparison masks.

Outputs:
  - <stem>_input.png: input image resized for display
  - <stem>_ground_truth_bw.png: label mask, black background / white rooftop
  - <stem>_prediction_bw.png: model prediction, black background / white rooftop
  - <stem>_comparison.png: OpenCV side-by-side input, ground truth, prediction

Example:
  python compare.py
  python compare.py --image dataset/archive/tiff/test/22828930_15.tiff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "UKAN" / "train_ukan.ipynb"
MODEL_PATH = PROJECT_ROOT / "UKAN" / "models" / "ukan_best.pth"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "archive"
DEFAULT_IMAGE_DIR = DATASET_ROOT / "tiff" / "test"
DEFAULT_LABEL_DIR = DATASET_ROOT / "tiff" / "test_labels"
OUTPUT_DIR = PROJECT_ROOT / "UKAN" / "comparison_outputs"

IMG_SIZE = 256
NUM_CLASSES = 2
BUILDING_CLASS_ID = 1


def load_ukan_class(notebook_path: Path):
    """Load the exact UKAN class definition from the training notebook."""
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Training notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    model_cell = None
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "class UKAN" in source and "class KANLinear" in source:
            model_cell = source
            break

    if model_cell is None:
        raise RuntimeError(f"Could not find the UKAN model definition in {notebook_path}")

    namespace = {
        "__name__": "ukan_from_notebook",
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
    }
    exec(model_cell, namespace)
    return namespace["UKAN"]


def first_image(image_dir: Path) -> Path:
    for pattern in ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"):
        matches = sorted(image_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No images found in {image_dir}")


def find_matching_label(image_path: Path, label_dir: Path | None) -> Path | None:
    if label_dir is None:
        return None
    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        candidate = label_dir / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    return image


def read_label_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"OpenCV could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    if size is not None:
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    return np.where(label > 127, 255, 0).astype(np.uint8)


def preprocess_for_model(bgr_image: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(bgr_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def predict_mask(model: nn.Module, bgr_image: np.ndarray, img_size: int, device: torch.device) -> np.ndarray:
    model_input = preprocess_for_model(bgr_image, img_size, device)
    with torch.no_grad():
        logits = model(model_input)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    return np.where(pred == BUILDING_CLASS_ID, 255, 0).astype(np.uint8)


def mask_metrics(pred_bw: np.ndarray, gt_bw: np.ndarray) -> dict[str, float]:
    pred = pred_bw > 127
    gt = gt_bw > 127
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-7
    return {
        "accuracy": float((tp + tn) / max(pred.size, 1)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
    }


def put_title(image: np.ndarray, title: str) -> np.ndarray:
    titled = image.copy()
    cv2.rectangle(titled, (0, 0), (titled.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        titled,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return titled


def save_comparison(
    image_bgr: np.ndarray,
    gt_bw: np.ndarray | None,
    pred_bw: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    display_size = (IMG_SIZE, IMG_SIZE)
    input_display = cv2.resize(image_bgr, display_size, interpolation=cv2.INTER_AREA)
    pred_display = cv2.cvtColor(pred_bw, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(output_dir / f"{stem}_input.png"), input_display)
    cv2.imwrite(str(output_dir / f"{stem}_prediction_bw.png"), pred_bw)

    panels = [put_title(input_display, "Input"), put_title(pred_display, "Prediction")]
    if gt_bw is not None:
        gt_display = cv2.cvtColor(gt_bw, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output_dir / f"{stem}_ground_truth_bw.png"), gt_bw)
        panels.insert(1, put_title(gt_display, "Ground truth"))

    comparison = cv2.hconcat(panels)
    cv2.imwrite(str(output_dir / f"{stem}_comparison.png"), comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UKAN rooftop prediction with ground truth label.")
    parser.add_argument("--image", type=Path, default=None, help="Input image path. Defaults to first TIFF test image.")
    parser.add_argument("--label", type=Path, default=None, help="Ground-truth label path. Auto-detected if omitted.")
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="Directory used to auto-detect labels.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to ukan_best.pth.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for output PNG files.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image is not None else first_image(DEFAULT_IMAGE_DIR)
    label_path = args.label if args.label is not None else find_matching_label(image_path, args.label_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    UKAN = load_ukan_class(NOTEBOOK_PATH)
    model = UKAN(num_classes=NUM_CLASSES, input_channels=3, img_size=IMG_SIZE, no_kan=False).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    image_bgr = read_color_image(image_path)
    pred_bw_256 = predict_mask(model, image_bgr, IMG_SIZE, device)

    gt_bw_256 = None
    if label_path is not None:
        gt_bw_256 = read_label_mask(label_path, size=(IMG_SIZE, IMG_SIZE))
        metrics = mask_metrics(pred_bw_256, gt_bw_256)
        print(
            "Metrics against ground truth "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"iou={metrics['iou']:.4f} "
            f"dice={metrics['dice']:.4f}"
        )
    else:
        print("No matching ground-truth label found; saving prediction only.")

    save_comparison(image_bgr, gt_bw_256, pred_bw_256, args.output_dir, image_path.stem)
    print(f"Input image: {image_path}")
    if label_path is not None:
        print(f"Ground truth: {label_path}")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Black/white rooftop prediction: {args.output_dir / (image_path.stem + '_prediction_bw.png')}")
    print(f"OpenCV comparison image: {args.output_dir / (image_path.stem + '_comparison.png')}")


if __name__ == "__main__":
    main()
