from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_screening.quality import assess_quality, quality_to_row, read_image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda items, **_: items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fundus image quality and recapture feedback.")
    parser.add_argument("--csv", default="train.csv", help="CSV with id_code column.")
    parser.add_argument("--image-dir", default="train_images", help="Directory containing image files.")
    parser.add_argument("--output", default="outputs/quality_report.csv", help="Output quality CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    if "id_code" not in df.columns:
        raise ValueError(f"{args.csv} must contain an id_code column.")
    if args.limit > 0:
        df = df.head(args.limit)

    rows: list[dict[str, object]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Assessing quality"):
        id_code = row.id_code
        image_path = Path(args.image_dir) / f"{id_code}.png"
        quality = assess_quality(read_image(image_path))
        output_row = quality_to_row(id_code, quality)
        if hasattr(row, "diagnosis"):
            output_row["diagnosis"] = int(row.diagnosis)
        rows.append(output_row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    print(f"Saved quality report: {output_path}")
    print(report["quality_status"].value_counts().to_string())


if __name__ == "__main__":
    main(parse_args())
