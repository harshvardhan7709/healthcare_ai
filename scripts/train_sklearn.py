from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.grading import (
    CLINICAL_MIN_SENSITIVITY,
    CLINICAL_MIN_SPECIFICITY,
    ICDR_SEVERITY_SCALE,
    choose_referable_threshold,
    referable_risk,
)
from dr_screening.features import extract_image_features, feature_names
from dr_screening.quality import QualityThresholds, quality_to_row

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda items, **_: items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a diabetic-retinopathy baseline with image quality gating and ICDR grading."
    )
    parser.add_argument("--csv", default="train.csv", help="Training CSV with id_code and diagnosis columns.")
    parser.add_argument("--image-dir", default="train_images", help="Directory containing PNG fundus images.")
    parser.add_argument("--output", default="outputs/sklearn_dr_model.joblib", help="Saved model bundle path.")
    parser.add_argument(
        "--quality-report",
        default="outputs/train_quality_report.csv",
        help="CSV path for quality metrics, decisions, and recapture feedback.",
    )
    parser.add_argument("--image-size", type=int, default=384, help="Preprocessed square image size.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick smoke tests.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest tree count.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for RandomForest. Use -1 on unrestricted machines.")
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="Thread count for parallel image feature extraction.",
    )
    parser.add_argument(
        "--feature-cache",
        default="",
        help="Optional path to cache/load extracted features to accelerate re-training.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Train on ungradeable images too. Default skips rejected images.",
    )
    return parser.parse_args()


def _limited_frame(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df
    pieces = []
    for _, group in df.groupby("diagnosis"):
        take = max(1, round(limit * len(group) / len(df)))
        pieces.append(group.sample(n=min(take, len(group)), random_state=seed))
    sampled = pd.concat(pieces).sample(frac=1.0, random_state=seed).head(limit)
    return sampled.reset_index(drop=True)


def _extract_single(task_tuple: tuple) -> tuple[np.ndarray | None, dict[str, object], int]:
    id_code, diagnosis, image_path, image_size, reject_ungradeable, thresholds = task_tuple
    features, quality = extract_image_features(
        image_path,
        image_size=image_size,
        reject_ungradeable=reject_ungradeable,
        enhance=True,
        thresholds=thresholds,
    )
    q_row = quality_to_row(id_code, quality)
    q_row["diagnosis"] = int(diagnosis)
    return features, q_row, int(diagnosis)


def build_feature_matrix(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if args.feature_cache and Path(args.feature_cache).exists() and args.limit <= 0:
        print(f"Loading cached features from: {args.feature_cache}")
        cache_data = joblib.load(args.feature_cache)
        return cache_data["x_values"], cache_data["y_values"], cache_data["quality_df"]

    df = pd.read_csv(args.csv)
    required = {"id_code", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing required columns: {sorted(missing)}")

    df = _limited_frame(df, args.limit, args.seed)
    image_dir = Path(args.image_dir)
    thresholds = QualityThresholds()
    reject_ungradeable = not args.include_rejected

    tasks = [
        (
            str(row.id_code),
            int(row.diagnosis),
            image_dir / f"{row.id_code}.png",
            args.image_size,
            reject_ungradeable,
            thresholds,
        )
        for row in df.itertuples(index=False)
    ]

    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    quality_rows: list[dict[str, object]] = []

    if args.extract_workers > 1:
        with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_extract_single, tasks),
                    total=len(tasks),
                    desc=f"Extracting features ({args.extract_workers} workers)",
                )
            )
    else:
        results = [
            _extract_single(t)
            for t in tqdm(tasks, total=len(tasks), desc="Extracting features")
        ]

    for feats, q_row, diag in results:
        quality_rows.append(q_row)
        if feats is not None:
            x_values.append(feats)
            y_values.append(diag)

    if not x_values:
        raise RuntimeError("No gradeable training images were available after quality filtering.")

    quality_df = pd.DataFrame(quality_rows)
    x_matrix = np.vstack(x_values)
    y_vector = np.asarray(y_values, dtype=np.int64)

    if args.feature_cache and args.limit <= 0:
        cache_path = Path(args.feature_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"x_values": x_matrix, "y_values": y_vector, "quality_df": quality_df}, cache_path)
        print(f"Saved feature cache: {cache_path}")

    return x_matrix, y_vector, quality_df


def train(args: argparse.Namespace) -> dict[str, object]:
    x_values, y_values, quality_df = build_feature_matrix(args)
    classes, class_counts = np.unique(y_values, return_counts=True)
    if len(classes) < 2:
        raise RuntimeError(f"Need at least two classes after filtering; got {classes.tolist()}")

    stratify = y_values if np.min(class_counts) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_values,
        y_values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    val_probabilities = model.predict_proba(x_val)
    referable_scores = np.asarray(
        [referable_risk(row, classes=model.classes_) for row in val_probabilities],
        dtype=np.float32,
    )
    referable_threshold = choose_referable_threshold(
        y_val,
        referable_scores,
        min_sensitivity=CLINICAL_MIN_SENSITIVITY,
        min_specificity=CLINICAL_MIN_SPECIFICITY,
    )

    y_val_ref = y_val >= 2
    y_ref_pred = referable_scores >= referable_threshold.threshold
    tn, fp, fn, tp = confusion_matrix(y_val_ref, y_ref_pred, labels=[False, True]).ravel()
    referable_sensitivity = float(tp / max(tp + fn, 1))
    referable_specificity = float(tn / max(tn + fp, 1))
    referable_accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))

    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_val, val_pred, weights="quadratic")),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_total_gradeable": int(len(y_values)),
        "referable_validation_target_sensitivity": CLINICAL_MIN_SENSITIVITY,
        "referable_validation_target_specificity": CLINICAL_MIN_SPECIFICITY,
        "referable_threshold": float(referable_threshold.threshold),
        "referable_validation_sensitivity": referable_sensitivity,
        "referable_validation_specificity": referable_specificity,
        "referable_validation_ppv": float(referable_threshold.positive_predictive_value),
        "referable_validation_npv": float(referable_threshold.negative_predictive_value),
        "referable_validation_f1": float(referable_threshold.f1_score),
        "referable_validation_roc_auc": float(referable_threshold.roc_auc),
        "referable_validation_youden_j": float(referable_threshold.youden_j),
        "referable_validation_accuracy": referable_accuracy,
        "referable_validation_positive_rate": float(referable_threshold.positive_rate),
        "referable_validation_met_target": bool(referable_threshold.met_target),
        "referable_confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "quality_status_counts": quality_df["quality_status"].value_counts().to_dict(),
        "class_counts_after_quality_filter": {
            str(cls): int(count) for cls, count in zip(classes.tolist(), class_counts.tolist())
        },
        "classification_report": classification_report(y_val, val_pred, zero_division=0, output_dict=True),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_names(),
        "image_size": args.image_size,
        "quality_thresholds": asdict(QualityThresholds()),
        "classes": classes.tolist(),
        "referable_level": 2,
        "referable_threshold": float(referable_threshold.threshold),
        "icdr_scale": {level: asdict(info) for level, info in ICDR_SEVERITY_SCALE.items()},
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, output_path)

    quality_path = Path(args.quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(quality_path, index=False)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {output_path}")
    print(f"Saved quality report: {quality_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        "Validation ICDR Performance: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
        f"qwk={metrics['quadratic_weighted_kappa']:.4f}"
    )
    print(
        "Referable DR (Level 2+) Clinical Benchmark: "
        f"sensitivity={referable_sensitivity:.2%} (target >90%), "
        f"specificity={referable_specificity:.2%} (target >85%), "
        f"roc_auc={metrics['referable_validation_roc_auc']:.4f}, "
        f"threshold={referable_threshold.threshold:.3f}, "
        f"met_clinical_target={referable_threshold.met_target}"
    )
    if not referable_threshold.met_target:
        print(
            "Note: The validation split reached "
            f"Sens={referable_sensitivity:.1%}, Spec={referable_specificity:.1%}. "
            "Safety priority retained to ensure referable cases are not missed."
        )
    return metrics


if __name__ == "__main__":
    train(parse_args())

