from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ACCEPTED = "accepted"
BORDERLINE = "borderline"
REJECTED = "rejected"


@dataclass
class QualityThresholds:
    """Transparent thresholds used by the heuristic quality gate."""

    min_fov_area_ratio: float = 0.18
    good_fov_area_ratio: float = 0.34
    min_focus_var: float = 8.0
    good_focus_var: float = 70.0
    min_contrast_std: float = 6.0
    good_contrast_std: float = 28.0
    min_brightness: float = 18.0
    good_low_brightness: float = 55.0
    good_high_brightness: float = 185.0
    max_brightness: float = 235.0
    good_illumination_cv: float = 0.18
    max_illumination_cv: float = 0.70
    max_center_offset: float = 0.38
    accept_score: float = 0.68
    reject_score: float = 0.35


@dataclass
class QualityResult:
    status: str
    score: float
    metrics: dict[str, float]
    reasons: list[str]
    feedback: list[str]

    @property
    def is_gradeable(self) -> bool:
        return self.status != REJECTED


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, preserving support for Windows paths."""

    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _threshold_score(value: float, fail: float, good: float) -> float:
    if good == fail:
        return 1.0 if value >= good else 0.0
    return _clamp((value - fail) / (good - fail))


def _range_score(value: float, low_fail: float, low_good: float, high_good: float, high_fail: float) -> float:
    if value < low_fail or value > high_fail:
        return 0.0
    if low_good <= value <= high_good:
        return 1.0
    if value < low_good:
        return _threshold_score(value, low_fail, low_good)
    return _clamp((high_fail - value) / (high_fail - high_good))


def fundus_mask(image_bgr: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
    """Return a binary mask for the largest non-background fundus field."""

    if image_bgr.ndim != 3:
        raise ValueError("Expected a BGR image with 3 channels")

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, raw = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    kernel_size = max(5, int(min(h, w) * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < h * w * min_area_ratio:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def crop_to_fundus(image_bgr: np.ndarray, mask: np.ndarray | None = None, pad_fraction: float = 0.04) -> np.ndarray:
    """Crop to the detected field of view with light padding."""

    if mask is None:
        mask = fundus_mask(image_bgr)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_bgr.copy()

    x, y, w, h = cv2.boundingRect(coords)
    pad = int(max(w, h) * pad_fraction)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_bgr.shape[1], x + w + pad)
    y1 = min(image_bgr.shape[0], y + h + pad)
    return image_bgr[y0:y1, x0:x1].copy()


def _inner_mask(mask: np.ndarray) -> np.ndarray:
    kernel_size = max(3, int(min(mask.shape[:2]) * 0.025))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel)
    if np.count_nonzero(eroded) == 0:
        return mask
    return eroded


def _tile_illumination_cv(gray: np.ndarray, mask: np.ndarray, grid_size: int = 4) -> float:
    means: list[float] = []
    h, w = gray.shape[:2]
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = int(row * h / grid_size)
            y1 = int((row + 1) * h / grid_size)
            x0 = int(col * w / grid_size)
            x1 = int((col + 1) * w / grid_size)
            tile_mask = mask[y0:y1, x0:x1] > 0
            if tile_mask.mean() < 0.20:
                continue
            means.append(float(gray[y0:y1, x0:x1][tile_mask].mean()))
    if len(means) < 2:
        return 1.0
    values = np.asarray(means, dtype=np.float32)
    return float(values.std() / max(values.mean(), 1.0))


def assess_quality(image_bgr: np.ndarray, thresholds: QualityThresholds | None = None) -> QualityResult:
    """Assess focus, illumination, contrast, and field-of-view adequacy."""

    thresholds = thresholds or QualityThresholds()
    h, w = image_bgr.shape[:2]
    mask = fundus_mask(image_bgr)
    reasons: list[str] = []
    feedback: list[str] = []

    if np.count_nonzero(mask) == 0:
        return QualityResult(
            status=REJECTED,
            score=0.0,
            metrics={
                "focus_var": 0.0,
                "brightness_mean": 0.0,
                "contrast_std": 0.0,
                "illumination_cv": 1.0,
                "fov_area_ratio": 0.0,
                "fov_bbox_fill": 0.0,
                "fov_center_offset": 1.0,
                "fov_aspect_ratio": 0.0,
            },
            reasons=["field_of_view_missing"],
            feedback=["Retake the image with the optic disc and macula inside the fundus field."],
        )

    crop = crop_to_fundus(image_bgr, mask)
    metric_image = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    metric_mask = fundus_mask(metric_image)
    metric_mask = _inner_mask(metric_mask)
    gray = cv2.cvtColor(metric_image, cv2.COLOR_BGR2GRAY)
    masked = gray[metric_mask > 0]
    if masked.size == 0:
        masked = gray.reshape(-1)

    focus_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(masked.mean())
    contrast_std = float(masked.std())
    illumination_cv = _tile_illumination_cv(gray, metric_mask)

    original_area = float(np.count_nonzero(mask))
    fov_area_ratio = original_area / float(h * w)
    coords = cv2.findNonZero(mask)
    x, y, bw, bh = cv2.boundingRect(coords)
    fov_bbox_fill = original_area / max(float(bw * bh), 1.0)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    fov_center_offset = float(np.hypot(cx - w / 2.0, cy - h / 2.0) / np.hypot(w, h))
    fov_aspect_ratio = float(bw / max(bh, 1))

    focus_score = _threshold_score(focus_var, thresholds.min_focus_var, thresholds.good_focus_var)
    contrast_score = _threshold_score(contrast_std, thresholds.min_contrast_std, thresholds.good_contrast_std)
    brightness_score = _range_score(
        brightness_mean,
        thresholds.min_brightness,
        thresholds.good_low_brightness,
        thresholds.good_high_brightness,
        thresholds.max_brightness,
    )
    illumination_score = 1.0 - _threshold_score(
        illumination_cv,
        thresholds.good_illumination_cv,
        thresholds.max_illumination_cv,
    )
    fov_area_score = _threshold_score(
        fov_area_ratio,
        thresholds.min_fov_area_ratio,
        thresholds.good_fov_area_ratio,
    )
    center_score = 1.0 - _threshold_score(
        fov_center_offset,
        0.12,
        thresholds.max_center_offset,
    )
    fill_score = _threshold_score(fov_bbox_fill, 0.55, 0.72)
    fov_score = float(np.mean([fov_area_score, center_score, fill_score]))

    score = float(
        0.32 * focus_score
        + 0.20 * contrast_score
        + 0.17 * illumination_score
        + 0.18 * fov_score
        + 0.13 * brightness_score
    )

    if focus_score < 0.45:
        reasons.append("low_focus")
        feedback.append("Retake with steadier alignment and refocus before capture.")
    if contrast_score < 0.45:
        reasons.append("low_contrast")
        feedback.append("Check media clarity and use adequate exposure before recapture.")
    if brightness_mean < thresholds.good_low_brightness:
        reasons.append("underexposed")
        feedback.append("Increase illumination or exposure and center the fundus field.")
    elif brightness_mean > thresholds.good_high_brightness:
        reasons.append("overexposed")
        feedback.append("Reduce illumination or exposure to avoid washed-out retinal detail.")
    if illumination_score < 0.45:
        reasons.append("uneven_illumination")
        feedback.append("Recenter the eye and avoid edge shadow or lens reflection.")
    if fov_score < 0.50:
        reasons.append("field_of_view_incomplete")
        feedback.append("Retake with more complete retinal field coverage.")

    catastrophic = (
        fov_area_ratio < thresholds.min_fov_area_ratio
        or focus_var < thresholds.min_focus_var
        or contrast_std < thresholds.min_contrast_std
        or brightness_mean < thresholds.min_brightness
        or brightness_mean > thresholds.max_brightness
        or fov_center_offset > thresholds.max_center_offset
    )

    if catastrophic or score < thresholds.reject_score:
        status = REJECTED
    elif score < thresholds.accept_score or reasons:
        status = BORDERLINE
    else:
        status = ACCEPTED

    metrics = {
        "focus_var": focus_var,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "illumination_cv": illumination_cv,
        "fov_area_ratio": fov_area_ratio,
        "fov_bbox_fill": fov_bbox_fill,
        "fov_center_offset": fov_center_offset,
        "fov_aspect_ratio": fov_aspect_ratio,
        "focus_score": focus_score,
        "contrast_score": contrast_score,
        "brightness_score": brightness_score,
        "illumination_score": illumination_score,
        "fov_score": fov_score,
    }

    return QualityResult(
        status=status,
        score=score,
        metrics=metrics,
        reasons=sorted(set(reasons)),
        feedback=list(dict.fromkeys(feedback)),
    )


def adaptive_enhance(image_bgr: np.ndarray, quality: QualityResult | None = None) -> np.ndarray:
    """Apply illumination normalization, CLAHE, and light denoising."""

    image = image_bgr.copy()
    h, w = image.shape[:2]
    sigma = max(h, w) / 30.0
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    image = cv2.addWeighted(image, 4.0, background, -4.0, 128.0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clip_limit = 2.0
    if quality and ("low_contrast" in quality.reasons or "underexposed" in quality.reasons):
        clip_limit = 3.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    image = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    if quality and quality.status == BORDERLINE:
        image = cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)
    return image


def preprocess_for_model(
    image_or_path: str | Path | np.ndarray,
    image_size: int = 512,
    reject_ungradeable: bool = True,
    enhance: bool = True,
    thresholds: QualityThresholds | None = None,
) -> tuple[np.ndarray | None, QualityResult]:
    """Quality-gate and preprocess a fundus image for model input."""

    if isinstance(image_or_path, (str, Path)):
        original = read_image(image_or_path)
    else:
        original = image_or_path

    quality = assess_quality(original, thresholds=thresholds)
    if reject_ungradeable and quality.status == REJECTED:
        return None, quality

    mask = fundus_mask(original)
    cropped = crop_to_fundus(original, mask)
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if enhance:
        resized = adaptive_enhance(resized, quality)
    return resized, quality


def quality_to_row(id_code: str, quality: QualityResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id_code": id_code,
        "quality_status": quality.status,
        "quality_score": round(quality.score, 6),
        "quality_reasons": ";".join(quality.reasons),
        "recapture_feedback": " ".join(quality.feedback),
    }
    row.update({key: round(value, 6) for key, value in quality.metrics.items()})
    return row
