from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .quality import adaptive_enhance, fundus_mask, preprocess_for_model


@dataclass
class LocalizationResult:
    x: int
    y: int
    radius: int
    confidence: float


@dataclass
class HemorrhageResult:
    label: str
    score: float
    area_ratio: float
    severity: int


@dataclass
class SegmentationResult:
    optic_disc: LocalizationResult
    fovea: LocalizationResult
    vessel_mask: np.ndarray
    microaneurysm_mask: np.ndarray
    exudate_mask: np.ndarray
    hemorrhage: HemorrhageResult
    neovascularization_score: float
    neovascularization_flag: bool
    overlays: dict[str, np.ndarray]
    metrics: dict[str, float]
    notes: list[str]


def _safe_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _central_crop(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_bgr.copy(), (0, 0)
    x, y, w, h = cv2.boundingRect(coords)
    pad = int(max(w, h) * 0.05)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_bgr.shape[1], x + w + pad)
    y1 = min(image_bgr.shape[0], y + h + pad)
    return image_bgr[y0:y1, x0:x1].copy(), (x0, y0)


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _enhanced_green(image_bgr: np.ndarray) -> np.ndarray:
    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(green)


def _filter_components(mask: np.ndarray, min_area: int, max_area: int, max_aspect: float = 4.5) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect <= max_aspect:
            filtered[labels == idx] = 255
    return filtered


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Approximate vessel skeletonization using morphological thinning."""

    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = cv2.erode(binary, element)
        if cv2.countNonZero(binary) == 0:
            break
    return skeleton


def _detect_bright_blob(gray: np.ndarray, roi_mask: np.ndarray, min_radius: int, max_radius: int) -> LocalizationResult:
    blurred = cv2.GaussianBlur(gray, (0, 0), 5)
    masked = blurred.copy()
    masked[roi_mask == 0] = 0
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked)
    cx, cy = max_loc

    circle_confidence = float(max_val / max(np.mean(masked[roi_mask > 0]) + 1e-6, 1.0))
    radius = int(max(min_radius, min(max_radius, min(gray.shape[:2]) // 10)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, radius // 2),
        param1=100,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None and len(circles[0]) > 0:
        circles = np.round(circles[0]).astype(int)
        best = max(circles, key=lambda c: float(blurred[c[1], c[0]]))
        cx, cy, radius = int(best[0]), int(best[1]), int(best[2])
        circle_confidence = min(1.0, float(blurred[cy, cx]) / max(float(np.mean(blurred[roi_mask > 0])) + 1e-6, 1.0))

    return LocalizationResult(cx, cy, radius, float(np.clip(circle_confidence, 0.0, 1.0)))


def locate_optic_disc(image_bgr: np.ndarray, fov_mask: np.ndarray) -> LocalizationResult:
    gray = _to_gray(image_bgr)
    masked_gray = gray.copy()
    masked_gray[fov_mask == 0] = 0
    h, w = gray.shape[:2]
    min_radius = max(12, int(min(h, w) * 0.02))
    max_radius = max(min_radius + 1, int(min(h, w) * 0.18))
    candidate = _detect_bright_blob(masked_gray, fov_mask, min_radius, max_radius)
    return candidate


def locate_fovea(image_bgr: np.ndarray, fov_mask: np.ndarray, optic_disc: LocalizationResult) -> LocalizationResult:
    gray = _to_gray(image_bgr)
    inv = 255 - gray
    roi = cv2.GaussianBlur(inv, (0, 0), 9)

    h, w = gray.shape[:2]
    focus_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.ellipse(
        focus_mask,
        center=(w // 2, h // 2),
        axes=(int(w * 0.22), int(h * 0.18)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )
    focus_mask = cv2.bitwise_and(focus_mask, fov_mask)

    # Prefer the darkest, vessel-poor point near the center after excluding the disc.
    exclusion = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(exclusion, (optic_disc.x, optic_disc.y), max(optic_disc.radius * 2, 20), 255, -1)
    search_mask = cv2.bitwise_and(focus_mask, cv2.bitwise_not(exclusion))
    if np.count_nonzero(search_mask) == 0:
        search_mask = fov_mask.copy()

    scores = roi.copy()
    scores[search_mask == 0] = 0
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)
    cx, cy = max_loc
    confidence = float(np.clip(max_val / max(np.mean(scores[search_mask > 0]) + 1e-6, 1.0), 0.0, 1.0))
    radius = max(8, int(min(h, w) * 0.035))
    return LocalizationResult(int(cx), int(cy), int(radius), confidence)


def segment_vessels(image_bgr: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    green = _enhanced_green(image_bgr)
    inverted = 255 - green
    vessels = np.zeros_like(green, dtype=np.uint8)

    for ksize in (9, 15, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(inverted, cv2.MORPH_BLACKHAT, kernel)
        vessels = cv2.max(vessels, blackhat)

    vessels = cv2.GaussianBlur(vessels, (3, 3), 0)
    _, vessel_mask = cv2.threshold(vessels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel_mask = cv2.bitwise_and(vessel_mask, fov_mask)

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return vessel_mask


def detect_microaneurysms(image_bgr: np.ndarray, fov_mask: np.ndarray, vessel_mask: np.ndarray, optic_disc: LocalizationResult) -> np.ndarray:
    green = _enhanced_green(image_bgr)
    dark = 255 - green
    response = cv2.GaussianBlur(dark, (0, 0), 1.5)
    response = cv2.GaussianBlur(response, (0, 0), 0.8)
    cutoff = np.percentile(response[fov_mask > 0], 99.25) if np.count_nonzero(fov_mask) else 255
    lesions = np.zeros_like(response, dtype=np.uint8)
    lesions[response >= cutoff] = 255
    lesions = cv2.bitwise_and(lesions, fov_mask)
    lesions = cv2.bitwise_and(lesions, cv2.bitwise_not(vessel_mask))

    exclusion = np.zeros_like(lesions)
    cv2.circle(exclusion, (optic_disc.x, optic_disc.y), max(optic_disc.radius * 2, 24), 255, -1)
    lesions = cv2.bitwise_and(lesions, cv2.bitwise_not(exclusion))

    mask = _filter_components(lesions, min_area=4, max_area=60, max_aspect=2.8)
    return mask


def segment_exudates(image_bgr: np.ndarray, fov_mask: np.ndarray, optic_disc: LocalizationResult) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, _, b_chan = cv2.split(lab)
    bright = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)
    yellow = cv2.normalize(b_chan, None, 0, 255, cv2.NORM_MINMAX)
    response = cv2.max(bright, yellow)
    response = cv2.GaussianBlur(response, (0, 0), 2.2)
    response = cv2.morphologyEx(response, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cutoff = np.percentile(response[fov_mask > 0], 99.15) if np.count_nonzero(fov_mask) else 255
    exudates = np.zeros_like(response, dtype=np.uint8)
    exudates[response >= cutoff] = 255
    exudates = cv2.bitwise_and(exudates, fov_mask)

    exclusion = np.zeros_like(exudates)
    cv2.circle(exclusion, (optic_disc.x, optic_disc.y), max(optic_disc.radius * 2, 26), 255, -1)
    exudates = cv2.bitwise_and(exudates, cv2.bitwise_not(exclusion))
    exudates = cv2.morphologyEx(
        exudates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return _filter_components(exudates, min_area=10, max_area=1200, max_aspect=5.0)


def classify_hemorrhage(image_bgr: np.ndarray, fov_mask: np.ndarray, microaneurysm_mask: np.ndarray) -> HemorrhageResult:
    green = _enhanced_green(image_bgr)
    dark = 255 - green
    candidate = cv2.GaussianBlur(dark, (0, 0), 2.0)
    candidate = cv2.normalize(candidate, None, 0, 255, cv2.NORM_MINMAX)
    cutoff = np.percentile(candidate[fov_mask > 0], 99.0) if np.count_nonzero(fov_mask) else 255
    hemorrhage_mask = np.zeros_like(candidate, dtype=np.uint8)
    hemorrhage_mask[candidate >= cutoff] = 255
    hemorrhage_mask = cv2.bitwise_and(hemorrhage_mask, fov_mask)
    hemorrhage_mask = cv2.bitwise_and(hemorrhage_mask, cv2.bitwise_not(microaneurysm_mask))
    hemorrhage_mask = cv2.morphologyEx(
        hemorrhage_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    hemorrhage_mask = _filter_components(hemorrhage_mask, min_area=20, max_area=1800, max_aspect=6.0)
    area_ratio = float(np.count_nonzero(hemorrhage_mask) / max(np.count_nonzero(fov_mask), 1))

    if area_ratio < 0.0025:
        severity = 0
        label = "none"
    elif area_ratio < 0.008:
        severity = 1
        label = "mild"
    elif area_ratio < 0.02:
        severity = 2
        label = "moderate"
    else:
        severity = 3
        label = "severe"

    score = float(np.clip(area_ratio / 0.03, 0.0, 1.0))
    return HemorrhageResult(label=label, score=score, area_ratio=area_ratio, severity=severity)


def detect_neovascularization(
    image_bgr: np.ndarray,
    fov_mask: np.ndarray,
    vessel_mask: np.ndarray,
    optic_disc: LocalizationResult,
) -> tuple[float, bool, dict[str, float]]:
    if np.count_nonzero(vessel_mask) == 0:
        return 0.0, False, {
            "vessel_density": 0.0,
            "branch_density": 0.0,
            "branch_to_vessel_ratio": 0.0,
            "disc_branch_density": 0.0,
        }

    skeleton = _skeletonize(vessel_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighborhood = cv2.filter2D((skeleton > 0).astype(np.float32), -1, kernel)
    branch_points = ((skeleton > 0) & (neighborhood >= 4)).astype(np.uint8)

    disc_radius = max(optic_disc.radius * 3, 36)
    branch_focus = np.zeros_like(vessel_mask)
    cv2.circle(branch_focus, (optic_disc.x, optic_disc.y), disc_radius, 255, -1)
    branch_near_disc = cv2.bitwise_and(branch_points * 255, branch_focus)

    vessel_density = float(np.count_nonzero(vessel_mask) / max(np.count_nonzero(fov_mask), 1))
    skeleton_density = float(np.count_nonzero(skeleton) / max(np.count_nonzero(fov_mask), 1))
    branch_density = float(np.count_nonzero(branch_points) / max(np.count_nonzero(fov_mask), 1))
    disc_branch_density = float(np.count_nonzero(branch_near_disc) / max(np.count_nonzero(branch_focus), 1))
    branch_to_vessel_ratio = float(branch_density / max(skeleton_density, 1e-6))
    score = float(
        np.clip(
            0.55 * branch_to_vessel_ratio
            + 0.45 * (disc_branch_density / 0.12),
            0.0,
            1.0,
        )
    )
    flag = score >= 0.80
    return score, flag, {
        "vessel_density": vessel_density,
        "skeleton_density": skeleton_density,
        "branch_density": branch_density,
        "branch_to_vessel_ratio": branch_to_vessel_ratio,
        "disc_branch_density": disc_branch_density,
    }


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = base_bgr.copy()
    color_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    color_layer[mask > 0] = color
    return cv2.addWeighted(color_layer, alpha, overlay, 1.0, 0.0)


def _draw_localization(image_bgr: np.ndarray, optic_disc: LocalizationResult, fovea: LocalizationResult) -> np.ndarray:
    overlay = image_bgr.copy()
    cv2.circle(overlay, (optic_disc.x, optic_disc.y), max(8, optic_disc.radius), (0, 255, 255), 2)
    cv2.circle(overlay, (optic_disc.x, optic_disc.y), 3, (0, 255, 255), -1)
    cv2.circle(overlay, (fovea.x, fovea.y), max(6, fovea.radius), (255, 0, 255), 2)
    cv2.circle(overlay, (fovea.x, fovea.y), 3, (255, 0, 255), -1)
    cv2.line(overlay, (optic_disc.x, optic_disc.y), (fovea.x, fovea.y), (255, 255, 0), 1)
    return overlay


def analyze_retinal_structures(image_or_path: str | Path | np.ndarray) -> SegmentationResult:
    processed, _ = preprocess_for_model(image_or_path, reject_ungradeable=False, enhance=True)
    if processed is None:
        raise ValueError("Could not preprocess image for segmentation.")

    fov_mask = fundus_mask(processed)
    if np.count_nonzero(fov_mask) == 0:
        raise ValueError("No fundus field found for segmentation.")

    crop, offset = _central_crop(processed, fov_mask)
    crop_mask = fundus_mask(crop)
    if np.count_nonzero(crop_mask) == 0:
        crop_mask = np.full(crop.shape[:2], 255, dtype=np.uint8)

    optic_disc = locate_optic_disc(crop, crop_mask)
    fovea = locate_fovea(crop, crop_mask, optic_disc)
    vessel_mask = segment_vessels(crop, crop_mask)
    microaneurysm_mask = detect_microaneurysms(crop, crop_mask, vessel_mask, optic_disc)
    exudate_mask = segment_exudates(crop, crop_mask, optic_disc)
    hemorrhage = classify_hemorrhage(crop, crop_mask, microaneurysm_mask)
    nv_score, nv_flag, nv_metrics = detect_neovascularization(crop, crop_mask, vessel_mask, optic_disc)

    vessel_overlay = _overlay_mask(crop, vessel_mask, (0, 200, 0))
    ma_overlay = _overlay_mask(crop, microaneurysm_mask, (0, 0, 255))
    exudate_overlay = _overlay_mask(crop, exudate_mask, (0, 255, 255))
    local_overlay = _draw_localization(crop, optic_disc, fovea)
    combined = local_overlay.copy()
    combined = _overlay_mask(combined, vessel_mask, (0, 180, 0), alpha=0.30)
    combined = _overlay_mask(combined, exudate_mask, (0, 255, 255), alpha=0.35)
    combined = _overlay_mask(combined, microaneurysm_mask, (0, 0, 255), alpha=0.45)

    metrics = {
        "optic_disc_x": float(optic_disc.x + offset[0]),
        "optic_disc_y": float(optic_disc.y + offset[1]),
        "optic_disc_radius": float(optic_disc.radius),
        "optic_disc_confidence": float(optic_disc.confidence),
        "fovea_x": float(fovea.x + offset[0]),
        "fovea_y": float(fovea.y + offset[1]),
        "fovea_radius": float(fovea.radius),
        "fovea_confidence": float(fovea.confidence),
        "vessel_coverage": float(np.count_nonzero(vessel_mask) / max(np.count_nonzero(crop_mask), 1)),
        "microaneurysm_area_ratio": float(np.count_nonzero(microaneurysm_mask) / max(np.count_nonzero(crop_mask), 1)),
        "exudate_area_ratio": float(np.count_nonzero(exudate_mask) / max(np.count_nonzero(crop_mask), 1)),
        "hemorrhage_area_ratio": float(hemorrhage.area_ratio),
        "hemorrhage_score": float(hemorrhage.score),
        "neovascularization_score": float(nv_score),
        "neovascularization_branch_to_vessel_ratio": float(nv_metrics["branch_to_vessel_ratio"]),
        "neovascularization_branch_density": float(nv_metrics["branch_density"]),
        "neovascularization_disc_branch_density": float(nv_metrics["disc_branch_density"]),
    }

    notes: list[str] = []
    if optic_disc.confidence < 0.35:
        notes.append("Optic disc localization is approximate.")
    if fovea.confidence < 0.30:
        notes.append("Fovea localization is approximate.")
    if metrics["vessel_coverage"] < 0.05:
        notes.append("Vessel segmentation is sparse; enhancement or recapture may help.")

    overlays = {
        "localized": _safe_uint8(local_overlay),
        "vessels": _safe_uint8(vessel_overlay),
        "microaneurysms": _safe_uint8(ma_overlay),
        "exudates": _safe_uint8(exudate_overlay),
        "combined": _safe_uint8(combined),
    }

    return SegmentationResult(
        optic_disc=optic_disc,
        fovea=fovea,
        vessel_mask=vessel_mask,
        microaneurysm_mask=microaneurysm_mask,
        exudate_mask=exudate_mask,
        hemorrhage=hemorrhage,
        neovascularization_score=nv_score,
        neovascularization_flag=nv_flag,
        overlays=overlays,
        metrics=metrics,
        notes=notes,
    )
