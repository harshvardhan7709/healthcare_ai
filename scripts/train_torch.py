from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

torch = None
nn = None
DataLoader = None
Dataset = object

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda items, **_: items

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.quality import QualityThresholds, preprocess_for_model, quality_to_row, read_image, write_image
from dr_screening.torch_model import build_model


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for diabetic-retinopathy grading.")
    parser.add_argument("--csv", default="train.csv", help="Training CSV with id_code and diagnosis.")
    parser.add_argument("--image-dir", default="train_images", help="Directory containing fundus images.")
    parser.add_argument("--output", default="outputs/torch_dr_model.pt", help="Checkpoint output path.")
    parser.add_argument("--quality-report", default="outputs/torch_quality_report.csv")
    parser.add_argument("--cache-dir", default="outputs/processed_train_images")
    parser.add_argument("--model", default="simple_cnn", choices=["simple_cnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights when available.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    parser.add_argument("--include-rejected", action="store_true", help="Do not skip rejected quality images.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. Use auto to prefer NVIDIA CUDA when available.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--multi-gpu", action="store_true", help="Wrap the model in DataParallel when multiple GPUs are visible.")
    return parser.parse_args()


class CachedFundusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_bgr = read_image(row.cache_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.train:
            if np.random.rand() < 0.5:
                image_rgb = cv2.flip(image_rgb, 1)
            if np.random.rand() < 0.35:
                alpha = float(np.random.uniform(0.90, 1.10))
                beta = float(np.random.uniform(-8.0, 8.0))
                image_rgb = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)

        image = image_rgb.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        label = int(row.diagnosis)
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def _limited_frame(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df
    pieces = []
    for _, group in df.groupby("diagnosis"):
        take = max(1, round(limit * len(group) / len(df)))
        pieces.append(group.sample(n=min(take, len(group)), random_state=seed))
    return pd.concat(pieces).sample(frac=1.0, random_state=seed).head(limit).reset_index(drop=True)


def build_processed_cache(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.csv)
    df = _limited_frame(df, args.limit, args.seed)
    image_dir = Path(args.image_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    thresholds = QualityThresholds()
    kept_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Quality gate/cache"):
        id_code = str(row.id_code)
        diagnosis = int(str(row.diagnosis))
        image_path = image_dir / f"{id_code}.png"
        processed, quality = preprocess_for_model(
            image_path,
            image_size=args.image_size,
            reject_ungradeable=not args.include_rejected,
            enhance=True,
            thresholds=thresholds,
        )
        quality_row = quality_to_row(id_code, quality)
        quality_row["diagnosis"] = diagnosis
        quality_rows.append(quality_row)
        if processed is None:
            continue

        cache_path = cache_dir / f"{row.id_code}.jpg"
        if not cache_path.exists():
            write_image(cache_path, processed)
        kept_rows.append(
            {
                "id_code": id_code,
                "diagnosis": diagnosis,
                "cache_path": str(cache_path),
                "quality_status": quality.status,
                "quality_score": quality.score,
            }
        )

    kept_df = pd.DataFrame(kept_rows)
    quality_df = pd.DataFrame(quality_rows)
    if kept_df.empty:
        raise RuntimeError("No images survived quality filtering.")
    return kept_df, quality_df


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def main(args: argparse.Namespace) -> None:
    global torch, nn, DataLoader, Dataset
    try:
        import torch as _torch
        from torch import nn as _nn
        from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise SystemExit(
            "PyTorch is not available in this Python environment. "
            "Install torch/torchvision for CNN training, or run scripts/train_sklearn.py now."
        ) from exc

    torch = _torch
    nn = _nn
    DataLoader = _DataLoader
    Dataset = _Dataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested with --device cuda, but PyTorch cannot see an NVIDIA GPU.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        print(f"Using NVIDIA CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU training")

    kept_df, quality_df = build_processed_cache(args)
    class_counts = kept_df["diagnosis"].value_counts().sort_index()
    stratify = kept_df["diagnosis"] if class_counts.min() >= 2 else None
    train_df, val_df = train_test_split(
        kept_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    train_loader = DataLoader(CachedFundusDataset(train_df, train=True),
    batch_size=args.batch_size,shuffle=True,num_workers=max(0, args.num_workers),
    pin_memory=device.type == "cuda", 
    persistent_workers=bool(args.num_workers and args.num_workers > 0))
    val_loader = DataLoader(
        CachedFundusDataset(val_df, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )

    model = build_model(args.model, num_classes=5, pretrained=args.pretrained).to(device)
    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Enabling DataParallel across {torch.cuda.device_count()} visible GPUs")
        model = nn.DataParallel(model)
    counts = np.bincount(train_df["diagnosis"].to_numpy(), minlength=5).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_qwk = -1.0
    best_metrics: dict[str, float] = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = running_loss / max(len(train_df), 1)
        print(
            f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, bal_acc={metrics['balanced_accuracy']:.4f}, "
            f"qwk={metrics['quadratic_weighted_kappa']:.4f}"
        )

        if metrics["quadratic_weighted_kappa"] > best_qwk:
            best_qwk = metrics["quadratic_weighted_kappa"]
            best_metrics = metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "image_size": args.image_size,
                    "device": str(device),
                    "class_counts": class_counts.to_dict(),
                    "metrics": metrics,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                output_path,
            )

    quality_path = Path(args.quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(quality_path, index=False)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {output_path}")
    print(f"Saved quality report: {quality_path}")
    print(f"Saved best metrics: {metrics_path}")


if __name__ == "__main__":
    main(parse_args())
