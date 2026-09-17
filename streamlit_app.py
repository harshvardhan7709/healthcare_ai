from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.grading import (
    CLINICAL_MIN_SENSITIVITY,
    CLINICAL_MIN_SPECIFICITY,
    DRGradeResult,
    ICDR_SEVERITY_SCALE,
    dr_level_label,
    get_icdr_info,
    grade_dr_case,
    referable_risk,
)
from dr_screening.features import extract_features_from_processed
from dr_screening.segmentation import analyze_retinal_structures
from dr_screening.quality import ACCEPTED, BORDERLINE, REJECTED, QualityThresholds, assess_quality, preprocess_for_model


DEFAULT_MODEL = Path("outputs/sklearn_dr_model.joblib")


def decode_upload(uploaded_file) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, f"Model file not found: {path}"
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # pragma: no cover
        return None, f"Could not load model bundle: {exc}"
    if bundle.get("model") is None:
        return None, f"The bundle at {path} does not contain a model object."
    return bundle, None


def badge_for_quality(status: str) -> tuple[str, str]:
    if status == ACCEPTED:
        return "Verified", "success"
    if status == BORDERLINE:
        return "Verified with caution", "warning"
    return "Needs recapture", "error"


def predict_grade(bundle, processed_bgr: np.ndarray, quality) -> DRGradeResult:
    if bundle is None:
        return grade_dr_case([])
    model = bundle["model"]
    referable_threshold = float(bundle.get("referable_threshold", 0.5))
    features = extract_features_from_processed(processed_bgr, quality)
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    return grade_dr_case(
        probabilities,
        classes=model.classes_,
        referable_threshold=referable_threshold,
    )


def render_image_result(
    index: int,
    uploaded_file,
    bundle,
    image_size: int,
    thresholds: QualityThresholds,
    show_structures: bool,
) -> dict[str, object]:
    image_bgr = decode_upload(uploaded_file)
    quality = assess_quality(image_bgr, thresholds=thresholds)
    processed_bgr, _ = preprocess_for_model(
        image_bgr,
        image_size=image_size,
        reject_ungradeable=False,
        enhance=True,
        thresholds=thresholds,
    )
    grade = predict_grade(bundle, processed_bgr, quality)

    label, variant = badge_for_quality(quality.status)
    title = f"{index + 1}. {uploaded_file.name}"

    st.subheader(title)
    if variant == "success":
        st.success(f"{label} for clinical grading")
    elif variant == "warning":
        st.warning(f"{label} - review capture conditions before confirming diagnosis")
    else:
        st.error("Needs recapture before clinical grading")

    left, right = st.columns(2)
    with left:
        st.caption("Original upload")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with right:
        st.caption("Enhanced preview (Illumination & Contrast normalized)")
        st.image(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.metric("Quality score", f"{quality.score:.3f}")
        st.metric("Focus", f"{quality.metrics['focus_var']:.1f}")
        st.metric("Contrast", f"{quality.metrics['contrast_std']:.1f}")
    with metric_b:
        st.metric("Brightness", f"{quality.metrics['brightness_mean']:.1f}")
        st.metric("FOV coverage", f"{quality.metrics['fov_area_ratio']:.3f}")
        st.metric("Illumination", f"{quality.metrics['illumination_cv']:.3f}")

    st.divider()
    st.markdown("### International Clinical DR Severity Grading")
    st.caption("Graded according to the International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale (Levels 0–4).")

    if grade.severity_level is not None:
        level = grade.severity_level
        info = get_icdr_info(level)

        # Severity Banner with clinical color scheme
        if level == 0:
            st.success(f"**Level 0 — {info.label} ({info.short_label})**")
        elif level == 1:
            st.info(f"**Level 1 — {info.label} ({info.short_label})**")
        elif level == 2:
            st.warning(f"**Level 2 — {info.label} ({info.short_label})** — Referable DR")
        elif level == 3:
            st.error(f"**Level 3 — {info.label} ({info.short_label})** — Severe Referable DR")
        else:
            st.error(f"**Level 4 — {info.label} ({info.short_label})** — Sight-Threatening Proliferative DR")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ICDR Severity", f"Level {level}: {grade.icdr_short}")
            st.metric("Model Confidence", f"{grade.confidence:.1%}" if grade.confidence else "N/A")
        with col2:
            ref_badge = "YES (Level 2+)" if grade.referable_dr else "NO (Non-referable)"
            st.metric("Referable DR", ref_badge)
            st.metric(
                "Referable Probability",
                f"{grade.referable_probability:.1%}" if grade.referable_probability is not None else "N/A",
            )
        with col3:
            st.metric(
                "Decision Threshold",
                f"{grade.referable_threshold:.1%}" if grade.referable_threshold is not None else "N/A",
            )
            st.metric(
                "Clinical Benchmark",
                "Sens >90% | Spec >85%",
            )

        # Risk indicator bar
        if grade.referable_probability is not None and grade.referable_threshold is not None:
            risk_val = min(max(grade.referable_probability, 0.0), 1.0)
            st.progress(risk_val, text=f"Referable DR Risk: {risk_val:.1%} (Cutoff: {grade.referable_threshold:.1%})")

        # Clinical Management Recommendations
        st.markdown(
            f"""
            **Clinical Hallmark Criteria:**  
            *{grade.findings}*  
            
            **Referral Urgency & Action:**  
            - **Timeline:** {grade.referral_urgency}  
            - **Recommended Care:** {grade.clinical_action}
            """
        )

        # Probability distribution over all 5 ICDR classes
        if grade.probabilities:
            with st.expander("ICDR Level Probability Distribution", expanded=False):
                prob_data = pd.DataFrame(
                    [
                        {
                            "Level": f"Level {cls}: {ICDR_SEVERITY_SCALE[cls].short_label}",
                            "Probability": prob,
                            "Referable": "Yes" if cls >= 2 else "No",
                        }
                        for cls, prob in grade.probabilities.items()
                    ]
                )
                st.dataframe(prob_data, use_container_width=True, hide_index=True)
                st.bar_chart(prob_data.set_index("Level")["Probability"])

        if grade.clinical_notes:
            for note in grade.clinical_notes:
                st.info(note)
    else:
        st.info("No model loaded or image ungradeable. DR severity grade unavailable.")

    result_row = {
        "file": uploaded_file.name,
        "quality_status": quality.status,
        "quality_score": round(quality.score, 4),
        "verified": quality.status != REJECTED,
        "severity_level": grade.severity_level,
        "severity_label": grade.severity_label,
        "icdr_short": grade.icdr_short,
        "confidence": round(grade.confidence, 4) if grade.confidence is not None else None,
        "referable_dr": grade.referable_dr,
        "referable_probability": round(grade.referable_probability, 4) if grade.referable_probability is not None else None,
        "referable_threshold": round(grade.referable_threshold, 4) if grade.referable_threshold is not None else None,
        "referral_urgency": grade.referral_urgency,
        "reasons": ", ".join(quality.reasons) if quality.reasons else "",
        "recapture_feedback": " ".join(quality.feedback) if quality.feedback else "",
    }
    st.dataframe(pd.DataFrame([result_row]), use_container_width=True, hide_index=True)

    if quality.reasons:
        st.write("Why it was flagged")
        st.write("; ".join(quality.reasons))
    if quality.feedback:
        st.write("Recapture guidance")
        for item in quality.feedback:
            st.write(f"- {item}")

    if show_structures:
        try:
            structures = analyze_retinal_structures(image_bgr)
        except Exception as exc:  # pragma: no cover
            st.error(f"Segmentation failed: {exc}")
            structures = None
    else:
        structures = None

    if structures is not None:
        st.divider()
        st.subheader("Retinal Structure Segmentation")
        seg_col1, seg_col2, seg_col3 = st.columns(3)
        with seg_col1:
            st.caption("Localization")
            st.image(cv2.cvtColor(structures.overlays["localized"], cv2.COLOR_BGR2RGB), use_container_width=True)
        with seg_col2:
            st.caption("Vessels and lesions")
            st.image(cv2.cvtColor(structures.overlays["combined"], cv2.COLOR_BGR2RGB), use_container_width=True)
        with seg_col3:
            st.caption("Mask summary")
            summary = np.zeros_like(structures.overlays["localized"])
            summary[structures.vessel_mask > 0] = (0, 180, 0)
            summary[structures.exudate_mask > 0] = (0, 255, 255)
            summary[structures.microaneurysm_mask > 0] = (0, 0, 255)
            st.image(cv2.cvtColor(summary, cv2.COLOR_BGR2RGB), use_container_width=True)

        structure_row = {
            "optic_disc": f"({int(structures.metrics['optic_disc_x'])}, {int(structures.metrics['optic_disc_y'])})",
            "fovea": f"({int(structures.metrics['fovea_x'])}, {int(structures.metrics['fovea_y'])})",
            "vessel_coverage": round(structures.metrics["vessel_coverage"], 4),
            "microaneurysm_area_ratio": round(structures.metrics["microaneurysm_area_ratio"], 4),
            "exudate_area_ratio": round(structures.metrics["exudate_area_ratio"], 4),
            "hemorrhage_class": structures.hemorrhage.label,
            "hemorrhage_score": round(structures.hemorrhage.score, 4),
            "neovascularization_score": round(structures.neovascularization_score, 4),
            "neovascularization_flag": bool(structures.neovascularization_flag),
        }
        st.dataframe(pd.DataFrame([structure_row]), use_container_width=True, hide_index=True)

        if structures.notes:
            st.info(" ".join(structures.notes))

    return result_row


def main() -> None:
    st.set_page_config(page_title="DR Clinical Grading & Verification", layout="wide")
    st.title("Diabetic Retinopathy Screening & Severity Grading")
    st.caption("Fundus image quality verification, International Clinical DR severity grading (Levels 0–4), and referable DR screening.")

    with st.sidebar:
        st.header("Screening Settings")
        model_path = st.text_input("Model bundle", value=str(DEFAULT_MODEL))
        image_size = st.select_slider("Processing size", options=[256, 320, 384, 448, 512], value=384)
        show_structures = st.toggle("Show retinal structure segmentation", value=True)
        show_json = st.toggle("Show raw JSON summary", value=False)
        thresholds = QualityThresholds()
        st.caption("Pre-screening quality gating (focus, illumination, contrast, FOV) is applied before grading.")

    bundle, load_error = load_model_bundle(model_path)
    if load_error:
        st.warning(load_error)
        st.info("The app can still verify image quality without a model bundle.")
        bundle = None
    elif bundle and "metrics" in bundle:
        metrics = bundle["metrics"]
        with st.sidebar:
            st.divider()
            st.subheader("Model Clinical Benchmarks")
            st.metric(
                "Referable Sensitivity (Level 2+)",
                f"{metrics.get('referable_validation_sensitivity', 0.0):.1%}",
                help="Target: >90% clinically acceptable sensitivity to minimize false negatives.",
            )
            st.metric(
                "Referable Specificity",
                f"{metrics.get('referable_validation_specificity', 0.0):.1%}",
                help="Target: >85% clinically acceptable specificity to minimize unnecessary referrals.",
            )
            st.metric("Multiclass QWK", f"{metrics.get('quadratic_weighted_kappa', 0.0):.3f}")
            st.metric("Referable ROC AUC", f"{metrics.get('referable_validation_roc_auc', 0.0):.3f}")

    uploads = st.file_uploader(
        "Upload retinal fundus images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploads:
        st.stop()

    results: list[dict[str, object]] = []
    for index, uploaded_file in enumerate(uploads):
        with st.expander(uploaded_file.name, expanded=index == 0):
            results.append(render_image_result(index, uploaded_file, bundle, image_size, thresholds, show_structures))

    st.divider()
    st.subheader("Batch Summary")
    report_df = pd.DataFrame(results)
    st.metric("Total images", len(report_df))
    st.metric("Verified", int(report_df["verified"].sum()))
    st.metric("Needs recapture", int((report_df["quality_status"] == REJECTED).sum()))
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download verification report",
        data=csv_bytes,
        file_name="dr_verification_report.csv",
        mime="text/csv",
    )

    if show_json:
        st.code(json.dumps(results, indent=2), language="json")


if __name__ == "__main__":
    main()

