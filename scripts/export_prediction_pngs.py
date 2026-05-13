"""Export curated mammogram/prediction panels as browser-ready PNG files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("outputs")
UI_PUBLIC = Path("../harmonia-defense-ui/public")
OUT_DIR = UI_PUBLIC / "predictions"

UPSCALE = 2
SIZE_OUT = 256 * UPSCALE
GT_COLOR = (220, 38, 38)
PRED_COLOR = (6, 182, 212)
ALPHA = 0.5


def _to_grayscale_rgb(arr_uint8: np.ndarray) -> np.ndarray:
    return np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)


def _overlay(base_rgb: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask_bool] = out[mask_bool] * (1 - alpha) + color_arr * alpha
    return out.astype(np.uint8)


def _upscale_and_save(arr_rgb: np.ndarray, path: Path) -> None:
    img = Image.fromarray(arr_rgb, mode="RGB")
    img = img.resize((SIZE_OUT, SIZE_OUT), resample=Image.NEAREST)
    img.save(path, format="PNG", optimize=True)


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr.squeeze(-1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected (H, W) or (H, W, 1), got {arr.shape}")


def main() -> None:
    with (UI_PUBLIC / "data/samples_curated.json").open(encoding="utf-8") as f:
        curated = json.load(f)
    sample_indices = [sample["sample_idx"] for sample in curated["samples"]]

    X_test = np.load(ROOT / "final_npy/X_test.npy", mmap_mode="r")
    masks_test = np.load(ROOT / "final_npy/masks_test.npy", mmap_mode="r")
    predictions = {
        "c1": np.load(ROOT / "segmentation/evaluation/test_predictions.npy", mmap_mode="r"),
        "f1": np.load(ROOT / "fl/f1_iid/test_predictions.npy", mmap_mode="r"),
        "f2": np.load(ROOT / "fl/f2_iid_phe/test_predictions.npy", mmap_mode="r"),
        "f3": np.load(ROOT / "fl/f3_noniid/test_predictions.npy", mmap_mode="r"),
        "f4": np.load(ROOT / "fl/f4_noniid_phe/test_predictions.npy", mmap_mode="r"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx in sample_indices:
        base = X_test[idx].squeeze(-1).astype(np.uint8)
        base_rgb = _to_grayscale_rgb(base)
        gt_mask = masks_test[idx].squeeze(-1) > 127

        _upscale_and_save(base_rgb, OUT_DIR / f"sample_{idx:03d}_input.png")
        _upscale_and_save(_overlay(base_rgb, gt_mask, GT_COLOR, ALPHA), OUT_DIR / f"sample_{idx:03d}_gt.png")

        for cfg_key, pred_arr in predictions.items():
            pred = _squeeze_hw(pred_arr[idx])
            pred_mask = pred > 0.5
            pred_overlay = _overlay(base_rgb, pred_mask, PRED_COLOR, ALPHA)
            _upscale_and_save(pred_overlay, OUT_DIR / f"sample_{idx:03d}_{cfg_key}.png")

    print(f"Exported {len(sample_indices) * 7} PNG files to {OUT_DIR}")


if __name__ == "__main__":
    main()
