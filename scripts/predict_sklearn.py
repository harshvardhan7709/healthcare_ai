from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.grading import dr_level_label, get_icdr_info, grade_dr_case, referable_risk
from dr_screening.features import extract_image_features
from dr_screening.quality import QualityThresholds, quality_to_row

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda items, **_: items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diabetic-retinopathy screening with ICDR grading and quality feedback."
    )
    parser.add_argument("--model", default="outputs/sklearn_dr_model.joblib", help="Saved joblib model bundle.")
    parser.add_argument("--csv", default="test.csv", help="CSV containing id_code.")
    parser.add_argument("--image-dir", default="test_images", help="Directory containing images.")
    parser.add_argument(
        "--output",
        default="outputs/screening_predictions.csv",
        help="Detailed screening report with quality status and feedback.",
    )
    parser.add_argument(
        "--submission-output",
        default="",
        help="Optional two-column id_code,diagnosis CSV for competition-style submission.",
    )
    parser.add_argument(
        "--predict-rejected",
        action="store_true",
        help="Still produce a diagnosis for rejected images. Default marks them for recapture.",
    )
    parser.add_argument(
        "--rejected-diagnosis",
        type=int,
        default=-1,
        help="Diagnosis value used when an image is rejected and --predict-rejected is not set.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model file not found: {model_path}. Train one first with scripts/train_sklearn.py "
            f"or point --model at an existing .joblib bundle."
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]
    image_size = int(bundle.get("image_size", 384))
    referable_threshold = float(bundle.get("referable_threshold", 0.5))
    df = pd.read_csv(args.csv)
    if "id_code" not in df.columns:
        raise ValueError(f"{args.csv} must contain an id_code column.")
    if args.limit > 0:
        df = df.head(args.limit)

    rows: list[dict[str, object]] = []
    submission_rows: list[dict[str, object]] = []
    thresholds = QualityThresholds()
    rejected = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Screening images"):
        id_code = row.id_code
        image_path = Path(args.image_dir) / f"{id_code}.png"
        features, quality = extract_image_features(
            image_path,
            image_size=image_size,
            reject_ungradeable=not args.predict_rejected,
            enhance=True,
            thresholds=thresholds,
        )

        if features is None:
            diagnosis = args.rejected_diagnosis
            info = get_icdr_info(diagnosis) if diagnosis >= 0 else None
            diagnosis_label = None if diagnosis < 0 else dr_level_label(diagnosis)
            severity_level = diagnosis if diagnosis >= 0 else None
            severity_label = diagnosis_label
            icdr_short = info.short_label if info else None
            icdr_findings = info.findings if info else None
            referable_probability = np.nan
            referable_dr = False
            referral_urgency = "Recapture image before clinical referral"
            clinical_action = "Recapture required; optical quality or illumination insufficient for clinical grading."
            decision = "recapture"
            confidence = np.nan
            rejected += 1
        else:
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            grade_res = grade_dr_case(
                probabilities,
                classes=model.classes_,
                referable_threshold=referable_threshold,
            )
            diagnosis = int(grade_res.severity_level) if grade_res.severity_level is not None else 0
            diagnosis_label = grade_res.severity_label
            severity_level = grade_res.severity_level
            severity_label = grade_res.severity_label
            icdr_short = grade_res.icdr_short
            icdr_findings = grade_res.findings
            confidence = grade_res.confidence
            referable_probability = grade_res.referable_probability
            referable_dr = grade_res.referable_dr
            referral_urgency = grade_res.referral_urgency
            clinical_action = grade_res.clinical_action
            decision = "recapture" if quality.status == "rejected" else "grade"
            if quality.status == "rejected":
                rejected += 1

        output_row = quality_to_row(id_code, quality)
        output_row.update(
            {
                "diagnosis": diagnosis,
                "diagnosis_label": diagnosis_label,
                "severity_level": severity_level,
                "severity_label": severity_label,
                "icdr_short": icdr_short,
                "icdr_findings": icdr_findings,
                "confidence": confidence,
                "referable_probability": referable_probability,
                "referable_threshold": referable_threshold,
                "referable_dr": referable_dr,
                "referral_urgency": referral_urgency,
                "clinical_action": clinical_action,
                "decision": decision,
            }
        )
        rows.append(output_row)
        submission_rows.append({"id_code": id_code, "diagnosis": diagnosis})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved screening report: {output_path}")
    print(f"Images marked for recapture: {rejected}/{len(df)}")

    if args.submission_output:
        submission_path = Path(args.submission_output)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(submission_rows).to_csv(submission_path, index=False)
        print(f"Saved submission CSV: {submission_path}")


if __name__ == "__main__":
    main(parse_args())
