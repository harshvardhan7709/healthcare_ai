from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


# International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale
@dataclass(frozen=True)
class ICDRLevelInfo:
    level: int
    label: str
    short_label: str
    findings: str
    referable: bool
    referral_urgency: str
    clinical_action: str


ICDR_SEVERITY_SCALE: dict[int, ICDRLevelInfo] = {
    0: ICDRLevelInfo(
        level=0,
        label="No apparent retinopathy",
        short_label="No DR",
        findings="No microaneurysms, retinal hemorrhages, or other vascular lesions present.",
        referable=False,
        referral_urgency="Routine annual rescreening (12 months)",
        clinical_action="Annual dilated retinal exam; continue systemic glycemic, lipid, and blood pressure control.",
    ),
    1: ICDRLevelInfo(
        level=1,
        label="Mild non-proliferative diabetic retinopathy",
        short_label="Mild NPDR",
        findings="Microaneurysms only; no exudates, venous beading, or retinal hemorrhages.",
        referable=False,
        referral_urgency="Rescreen in 6–12 months",
        clinical_action="Routine follow-up in 6–12 months; intensify diabetic management and cardiovascular risk monitoring.",
    ),
    2: ICDRLevelInfo(
        level=2,
        label="Moderate non-proliferative diabetic retinopathy",
        short_label="Moderate NPDR",
        findings="More than microaneurysms but less than severe NPDR; dot/blot hemorrhages, hard exudates, or cotton wool spots.",
        referable=True,
        referral_urgency="Ophthalmic review within 1–3 months",
        clinical_action="Refer to an ophthalmologist for comprehensive evaluation and macular edema OCT assessment.",
    ),
    3: ICDRLevelInfo(
        level=3,
        label="Severe non-proliferative diabetic retinopathy",
        short_label="Severe NPDR",
        findings="Any 1 of the 4-2-1 criteria without signs of PDR: >20 intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, prominent IRMA in 1+ quadrant.",
        referable=True,
        referral_urgency="Prompt ophthalmic referral within 2–4 weeks",
        clinical_action="High risk of rapid progression to proliferative disease; prompt specialist assessment for retinal photocoagulation or anti-VEGF therapy.",
    ),
    4: ICDRLevelInfo(
        level=4,
        label="Proliferative diabetic retinopathy",
        short_label="PDR",
        findings="Neovascularization of the disc (NVD) or elsewhere (NVE), fibrous proliferation, and/or preretinal or vitreous hemorrhage.",
        referable=True,
        referral_urgency="Urgent ophthalmic referral within 24–48 hours",
        clinical_action="Sight-threatening emergency; urgent vitreo-retinal evaluation for panretinal photocoagulation (PRP), anti-VEGF, or vitrectomy.",
    ),
}

# Legacy mappings maintained for backwards compatibility
DR_LEVEL_LABELS: dict[int, str] = {
    level: info.label for level, info in ICDR_SEVERITY_SCALE.items()
}

REFERABLE_LEVEL: int = 2
CLINICAL_MIN_SENSITIVITY: float = 0.90
CLINICAL_MIN_SPECIFICITY: float = 0.85


@dataclass
class ReferableThresholdResult:
    threshold: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float
    f1_score: float
    balanced_accuracy: float
    youden_j: float
    positive_rate: float
    met_target: bool
    tp: int
    fp: int
    tn: int
    fn: int
    roc_auc: float


@dataclass
class DRGradeResult:
    severity_level: int | None
    severity_label: str | None
    icdr_short: str | None
    findings: str | None
    confidence: float | None
    probabilities: dict[int, float]
    referable_dr: bool | None
    referable_probability: float | None
    referable_threshold: float | None
    referral_urgency: str | None
    clinical_action: str | None
    clinical_notes: list[str]


def get_icdr_info(level: int | None) -> ICDRLevelInfo | None:
    if level is None:
        return None
    return ICDR_SEVERITY_SCALE.get(int(level))


def dr_level_label(level: int | None) -> str | None:
    if level is None:
        return None
    info = get_icdr_info(level)
    return info.label if info else f"Level {level}"


def referable_from_level(level: int) -> bool:
    return int(level) >= REFERABLE_LEVEL


def referable_risk(
    probabilities: Iterable[float],
    classes: Iterable[int] | None = None,
    referable_level: int = REFERABLE_LEVEL,
) -> float:
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if probs.size == 0:
        return 0.0
    if classes is None:
        return float(np.clip(probs[max(0, min(referable_level, probs.size - 1)) :].sum(), 0.0, 1.0))

    class_labels = np.asarray(list(classes), dtype=int)
    if class_labels.size != probs.size:
        raise ValueError("classes and probabilities must have the same length")
    return float(np.clip(probs[class_labels >= referable_level].sum(), 0.0, 1.0))


def choose_referable_threshold(
    y_true: np.ndarray,
    referable_scores: np.ndarray,
    min_sensitivity: float = CLINICAL_MIN_SENSITIVITY,
    min_specificity: float = CLINICAL_MIN_SPECIFICITY,
) -> ReferableThresholdResult:
    """Calibrate decision threshold to satisfy clinical sensitivity and specificity benchmarks.
    
    The International Clinical screening guidelines specify Sensitivity > 90%
    and Specificity > 85% for Referable DR (Level 2+).
    """
    y_true = np.asarray(y_true).astype(int)
    y_ref = y_true >= REFERABLE_LEVEL
    scores = np.asarray(referable_scores, dtype=np.float32)

    if scores.size == 0 or len(np.unique(y_ref)) < 2:
        return ReferableThresholdResult(
            threshold=0.5,
            sensitivity=0.0,
            specificity=0.0,
            positive_predictive_value=0.0,
            negative_predictive_value=0.0,
            f1_score=0.0,
            balanced_accuracy=0.0,
            youden_j=-1.0,
            positive_rate=0.0,
            met_target=False,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            roc_auc=0.5,
        )

    try:
        auc_val = float(roc_auc_score(y_ref, scores))
    except Exception:
        auc_val = 0.5

    # Evaluate across fine-grained empirical thresholds
    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0 if len(unique_scores) > 1 else np.array([])
    thresholds = np.unique(np.concatenate(([0.0], unique_scores, midpoints, [1.0])))

    candidates: list[ReferableThresholdResult] = []
    for threshold in thresholds:
        y_pred = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y_ref, y_pred, labels=[False, True]).ravel()
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        ppv = float(tp / max(tp + fp, 1))
        npv = float(tn / max(tn + fn, 1))
        f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
        bal_acc = float((sens + spec) / 2.0)
        youden_j = float(sens + spec - 1.0)
        pos_rate = float(np.mean(y_pred))
        met_target = bool(sens >= min_sensitivity and spec >= min_specificity)

        candidates.append(
            ReferableThresholdResult(
                threshold=float(threshold),
                sensitivity=sens,
                specificity=spec,
                positive_predictive_value=ppv,
                negative_predictive_value=npv,
                f1_score=f1,
                balanced_accuracy=bal_acc,
                youden_j=youden_j,
                positive_rate=pos_rate,
                met_target=met_target,
                tp=int(tp),
                fp=int(fp),
                tn=int(tn),
                fn=int(fn),
                roc_auc=auc_val,
            )
        )

    # 1. Candidates meeting BOTH clinical requirements (sens >= 90% AND spec >= 85%)
    qualified = [c for c in candidates if c.met_target]
    if qualified:
        # Choose candidate maximizing Youden's J while keeping sensitivity safe
        # Tie-breaker prefers higher sensitivity, then balanced accuracy
        qualified.sort(
            key=lambda c: (c.youden_j, c.sensitivity, c.balanced_accuracy),
            reverse=True,
        )
        return qualified[0]

    # 2. Clinical safety priority: Guarantee sensitivity >= min_sensitivity (90%), then maximize specificity
    sens_qualified = [c for c in candidates if c.sensitivity >= min_sensitivity]
    if sens_qualified:
        sens_qualified.sort(
            key=lambda c: (c.specificity, c.youden_j, c.sensitivity),
            reverse=True,
        )
        return sens_qualified[0]

    # 3. Fallback: minimize distance to target point in ROC space with 2x penalty on missed sensitivity
    def distance_penalty(c: ReferableThresholdResult) -> float:
        sens_gap = max(0.0, min_sensitivity - c.sensitivity)
        spec_gap = max(0.0, min_specificity - c.specificity)
        return (sens_gap * 2.0) ** 2 + spec_gap**2

    candidates.sort(key=distance_penalty)
    return candidates[0]


def grade_dr_case(
    probabilities: np.ndarray | list[float],
    classes: list[int] | np.ndarray | None = None,
    referable_threshold: float = 0.5,
) -> DRGradeResult:
    """Produce comprehensive ICDR severity grading and referable DR decision."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim > 1:
        probs = probs.squeeze()

    if classes is None:
        classes = list(range(len(probs)))
    else:
        classes = [int(c) for c in classes]

    if probs.size == 0 or len(classes) != probs.size:
        return DRGradeResult(
            severity_level=None,
            severity_label=None,
            icdr_short=None,
            findings=None,
            confidence=None,
            probabilities={},
            referable_dr=None,
            referable_probability=None,
            referable_threshold=referable_threshold,
            referral_urgency=None,
            clinical_action=None,
            clinical_notes=["Unable to compute severity grading due to invalid probabilities."],
        )

    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}
    top_idx = int(np.argmax(probs))
    pred_level = int(classes[top_idx])
    confidence = float(probs[top_idx])

    ref_prob = referable_risk(probs, classes=classes, referable_level=REFERABLE_LEVEL)
    is_referable = bool(ref_prob >= referable_threshold)

    info = get_icdr_info(pred_level)
    notes: list[str] = []

    if is_referable and pred_level < REFERABLE_LEVEL:
        notes.append(
            f"Note: Classified as Level {pred_level} ({info.short_label if info else ''}), "
            f"but elevated referable DR risk ({ref_prob:.1%}) exceeds calibrated threshold ({referable_threshold:.1%}). "
            "Clinical evaluation recommended."
        )

    return DRGradeResult(
        severity_level=pred_level,
        severity_label=info.label if info else f"Level {pred_level}",
        icdr_short=info.short_label if info else f"L{pred_level}",
        findings=info.findings if info else None,
        confidence=confidence,
        probabilities=prob_dict,
        referable_dr=is_referable,
        referable_probability=ref_prob,
        referable_threshold=float(referable_threshold),
        referral_urgency=info.referral_urgency if info else None,
        clinical_action=info.clinical_action if info else None,
        clinical_notes=notes,
    )
