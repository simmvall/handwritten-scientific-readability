from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytesseract

try:
    from src.document_readability import compute_cer, ocr_text_and_confidence
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.document_readability import compute_cer, ocr_text_and_confidence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "expanded_dataset"
IMAGE_DIR = EXPERIMENT / "images"
GROUND_TRUTH_DIR = EXPERIMENT / "ground_truth_provided"
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def correlation(x: pd.Series, y: pd.Series) -> Optional[float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    return float(np.corrcoef(valid.iloc[:, 0], valid.iloc[:, 1])[0, 1])


def rank_values(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ocr(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for row in metrics.to_dict(orient="records"):
        page_id = row["page_id"]
        image_path = IMAGE_DIR / row["image_filename"]
        truth_path = GROUND_TRUTH_DIR / f"{page_id}.txt"
        try:
            ocr_text, confidence = ocr_text_and_confidence(str(image_path))
            ground_truth = (
                truth_path.read_text(encoding="utf-8").strip()
                if truth_path.exists()
                else ""
            )
            rows.append(
                {
                    "page_id": page_id,
                    "notebook_id": row["notebook_id"],
                    "ocr_text": ocr_text,
                    "ocr_confidence": confidence,
                    "ground_truth_text": ground_truth,
                    "cer": compute_cer(ground_truth, ocr_text)
                    if truth_path.exists()
                    else None,
                    "ground_truth_status": row["ground_truth_status"],
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "page_id": page_id,
                    "notebook_id": row["notebook_id"],
                    "ocr_text": "",
                    "ocr_confidence": None,
                    "ground_truth_text": "",
                    "cer": None,
                    "ground_truth_status": row["ground_truth_status"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def correlation_rows(data: pd.DataFrame) -> List[dict]:
    targets = ["cer", "ocr_confidence"]
    metrics = [
        "baseline_consistency",
        "spatial_organization",
        "text_fragmentation",
        "combined_score",
    ]
    rows: List[dict] = []
    for target in targets:
        for metric in metrics:
            valid = data[[metric, target]].dropna()
            rows.extend(
                [
                    {
                        "target": target,
                        "metric": metric,
                        "method": "pearson",
                        "n": len(valid),
                        "correlation": correlation(valid[metric], valid[target]),
                    },
                    {
                        "target": target,
                        "metric": metric,
                        "method": "spearman",
                        "n": len(valid),
                        "correlation": correlation(
                            rank_values(valid[metric]), rank_values(valid[target])
                        ),
                    },
                ]
            )
    return rows


def main() -> None:
    tesseract_cmd = os.environ.get("TESSERACT_CMD", DEFAULT_TESSERACT_CMD)
    if not Path(tesseract_cmd).exists():
        raise FileNotFoundError(
            "Tesseract executable not found. Set TESSERACT_CMD or install it at "
            f"{DEFAULT_TESSERACT_CMD}"
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    metrics = pd.read_csv(EXPERIMENT / "structural_metrics.csv")
    ocr = run_ocr(metrics)
    combined = metrics.merge(ocr, on=["page_id", "notebook_id", "ground_truth_status"])
    combined.to_csv(EXPERIMENT / "page_level_results.csv", index=False)
    ocr.to_csv(EXPERIMENT / "ocr_cer_results.csv", index=False)
    per_notebook_dir = EXPERIMENT / "per_notebook"
    per_notebook_dir.mkdir(parents=True, exist_ok=True)
    for notebook_id, notebook_rows in combined.groupby("notebook_id", sort=True):
        notebook_rows.to_csv(
            per_notebook_dir / f"{notebook_id}_results.csv",
            index=False,
        )

    correlations = correlation_rows(combined)
    write_rows(EXPERIMENT / "correlations.csv", correlations)

    numeric = [
        "baseline_consistency",
        "spatial_organization",
        "text_fragmentation",
        "combined_score",
        "ocr_confidence",
        "cer",
    ]
    descriptive = (
        combined[numeric]
        .describe()
        .T.rename(
            columns={
                "count": "n",
                "mean": "mean",
                "std": "std",
                "min": "min",
                "max": "max",
            }
        )
    )
    descriptive["range"] = descriptive["max"] - descriptive["min"]
    descriptive[["n", "mean", "std", "min", "max", "range"]].to_csv(
        EXPERIMENT / "descriptive_statistics.csv"
    )

    weights = {
        "existing_0.40_0.20_0.40": (0.40, 0.20, 0.40),
        "equal_0.33_0.33_0.34": (0.33, 0.33, 0.34),
        "baseline_emphasis_0.50_0.20_0.30": (0.50, 0.20, 0.30),
        "fragmentation_emphasis_0.30_0.20_0.50": (0.30, 0.20, 0.50),
    }
    sensitivity: List[dict] = []
    for name, (wb, ws, wf) in weights.items():
        score = (
            wb * combined["baseline_consistency"]
            + ws * combined["spatial_organization"]
            + wf * combined["text_fragmentation"]
        )
        valid = pd.DataFrame({"score": score, "cer": combined["cer"]}).dropna()
        sensitivity.append(
            {
                "weighting": name,
                "baseline_weight": wb,
                "spatial_weight": ws,
                "fragmentation_weight": wf,
                "cer_n": len(valid),
                "pearson_score_vs_cer": correlation(valid["score"], valid["cer"]),
                "spearman_score_vs_cer": correlation(
                    rank_values(valid["score"]), rank_values(valid["cer"])
                ),
                "score_mean_all_pages": float(score.mean()),
                "score_min_all_pages": float(score.min()),
                "score_max_all_pages": float(score.max()),
            }
        )
    write_rows(EXPERIMENT / "weight_sensitivity.csv", sensitivity)

    notebook_summary = (
        combined.groupby("notebook_id")
        .agg(
            pages=("page_id", "count"),
            cer_pages=("cer", "count"),
            mean_combined_score=("combined_score", "mean"),
            mean_ocr_confidence=("ocr_confidence", "mean"),
            mean_cer=("cer", "mean"),
        )
        .reset_index()
    )
    notebook_summary.to_csv(EXPERIMENT / "notebook_summary.csv", index=False)

    config = {
        "dataset": "expanded_dataset",
        "pages": int(len(combined)),
        "notebooks": sorted(combined["notebook_id"].unique().tolist()),
        "author_verified_cer_pages": int(combined["cer"].notna().sum()),
        "pages_without_ground_truth": int(combined["cer"].isna().sum()),
        "ocr_engine": "Tesseract through pytesseract.image_to_data",
        "cer_normalization": (
            "Existing compute_cer implementation: JiWER when installed; "
            "fallback collapses whitespace before Levenshtein distance."
        ),
        "structural_method": "Existing src.document_readability implementation unchanged",
        "render_dpi": 300,
        "python_version": platform.python_version(),
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "opencv-python", "pytesseract", "jiwer")
            if _installed(package)
        },
        "tesseract_version": tesseract_version(tesseract_cmd),
    }
    (EXPERIMENT / "run_metadata.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def _installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def tesseract_version(command: str) -> str:
    result = subprocess.run(
        [command, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


if __name__ == "__main__":
    main()
