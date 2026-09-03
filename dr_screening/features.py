from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .quality import QualityResult, QualityThresholds, fundus_mask, preprocess_for_model


QUALITY_METRIC_NAMES = [
    "focus_var",
    "brightness_mean",
    "contrast_std",
    "illumination_cv",
    "fov_area_ratio",
    "fov_bbox_fill",
    "fov_center_offset",
    "fov_aspect_ratio",
    "focus_score",
    "contrast_score",
    "brightness_score",
    "illumination_score",
    "fov_score",
]

CHANNELS = [
    ("bgr_b", "bgr", 0),
    ("bgr_g", "bgr", 1),
    ("bgr_r", "bgr", 2),
    ("hsv_h", "hsv", 0),
    ("hsv_s", "hsv", 1),
    ("hsv_v", "hsv", 2),
    ("lab_l", "lab", 0),
    ("lab_a", "lab", 1),
    ("lab_b", "lab", 2),
]
SUMMARY_STATS = ["mean", "std", "p10", "p50", "p90"]


def feature_names() -> list[str]:
    names = [f"quality_{name}" for name in QUALITY_METRIC_NAMES]
    for prefix, _, _ in CHANNELS:
        names.extend(f"{prefix}_{stat}" for stat in SUMMARY_STATS)
    names.extend(f"green_hist_{idx:02d}" for idx in range(16))
    names.extend(f"hue_hist_{idx:02d}" for idx in range(12))
    names.extend(f"saturation_hist_{idx:02d}" for idx in range(8))
    names.extend(
        [
            "texture_laplacian_var",
            "texture_sobel_mean",
            "texture_sobel_std",
            "texture_sobel_p90",
            "texture_canny_density",
            "proxy_dark_lesion_ratio",
            "proxy_bright_lesion_ratio",
            "proxy_high_saturation_ratio",
        ]
    )
    return names


def _masked_channel_values(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = channel[mask > 0]
    if values.size == 0:
        values = channel.reshape(-1)
    return values.astype(np.float32)


def _summary(values: np.ndarray) -> list[float]:
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    ]


def _normalized_hist(values: np.ndarray, bins: int, value_range: tuple[int, int]) -> list[float]:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return [0.0] * bins
    return (hist / total).astype(float).tolist()


def extract_features_from_processed(image_bgr: np.ndarray, quality: QualityResult) -> np.ndarray:
    """Extract deterministic color, texture, and quality features."""

    mask = fundus_mask(image_bgr)
    if np.count_nonzero(mask) == 0:
        mask = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    spaces = {"bgr": image_bgr, "hsv": hsv, "lab": lab}

    values: list[float] = []
    values.extend(float(quality.metrics.get(name, 0.0)) for name in QUALITY_METRIC_NAMES)

    for _, space_name, channel_idx in CHANNELS:
        channel_values = _masked_channel_values(spaces[space_name][:, :, channel_idx], mask)
        values.extend(_summary(channel_values))

    green = _masked_channel_values(image_bgr[:, :, 1], mask)
    hue = _masked_channel_values(hsv[:, :, 0], mask)
    saturation = _masked_channel_values(hsv[:, :, 1], mask)
    values.extend(_normalized_hist(green, bins=16, value_range=(0, 256)))
    values.extend(_normalized_hist(hue, bins=12, value_range=(0, 180)))
    values.extend(_normalized_hist(saturation, bins=8, value_range=(0, 256)))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_values = _masked_channel_values(sobel_mag, mask)

    median_gray = float(np.median(gray[mask > 0])) if np.count_nonzero(mask) else float(np.median(gray))
    low = int(max(0, 0.66 * median_gray))
    high = int(min(255, 1.33 * median_gray + 20))
    edges = cv2.Canny(gray, low, high)
    canny_density = float(np.count_nonzero(edges[mask > 0]) / max(np.count_nonzero(mask), 1))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(image_bgr[:, :, 1])
    green_eq_values = _masked_channel_values(green_eq, mask)
    dark_lesion_ratio = float(np.mean(green_eq_values < 35))
    bright_lesion_ratio = float(np.mean(green_eq_values > 220))
    high_saturation_ratio = float(np.mean(saturation > 180))

    values.extend(
        [
            laplacian_var,
            float(sobel_values.mean()),
            float(sobel_values.std()),
            float(np.percentile(sobel_values, 90)),
            canny_density,
            dark_lesion_ratio,
            bright_lesion_ratio,
            high_saturation_ratio,
        ]
    )
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_image_features(
    image_or_path: str | Path | np.ndarray,
    image_size: int = 384,
    reject_ungradeable: bool = True,
    enhance: bool = True,
    thresholds: QualityThresholds | None = None,
) -> tuple[np.ndarray | None, QualityResult]:
    processed, quality = preprocess_for_model(
        image_or_path,
        image_size=image_size,
        reject_ungradeable=reject_ungradeable,
        enhance=enhance,
        thresholds=thresholds,
    )
    if processed is None:
        return None, quality
    return extract_features_from_processed(processed, quality), quality
