from __future__ import annotations

import csv
import difflib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

try:
    from src.document_readability import (
        _ensure_cv2,
        _load_image,
        compute_cer,
        preprocess_image,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.document_readability import (
        _ensure_cv2,
        _load_image,
        compute_cer,
        preprocess_image,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "expanded_dataset"
AUDIT = EXPERIMENT / "cer_fragmentation_audit"
IMAGE_DIR = EXPERIMENT / "images"
TRUTH_DIR = EXPERIMENT / "ground_truth_provided"
PAGE_RESULTS = EXPERIMENT / "page_level_results.csv"
MANIFEST = EXPERIMENT / "dataset_manifest.csv"
AUDIT_DPI = 300
SMALL_AREA = 150
MIN_AREA = 15
DIAGNOSTIC_PAGES = {23, 61, 79, 82}


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def exact_cer(reference: str, hypothesis: str) -> float:
    return levenshtein(reference, hypothesis) / max(1, len(reference))


def normalize_for_audit(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def strip_form_labels(text: str) -> str:
    """Conservative heuristic; retained separately from the verified CER."""
    value = text
    patterns = [
        r"\bcontinued\s+from\s+page\b",
        r"\bread\s+and\s+understood\s+by\b",
        r"\bnotebook\s+no\.?\b",
        r"\bproject\b",
        r"\bsigned\b",
        r"\bdate\b",
    ]
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return normalize_for_audit(value)


def excerpt_diff(reference: str, hypothesis: str, limit: int = 1200) -> str:
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    parts: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            parts.append(
                f"{tag}: ref={reference[i1:i2]!r} | ocr={hypothesis[j1:j2]!r}"
            )
        if sum(len(part) for part in parts) >= limit:
            break
    return " || ".join(parts)[:limit]


def audit_cer() -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(PAGE_RESULTS)
    alyssa = results[results["notebook_id"] == "alyssa_bordner_i_1"].copy()
    rows: List[dict] = []
    mapping_rows: List[dict] = []
    for row in alyssa.to_dict("records"):
        page_id = row["page_id"]
        truth_path = TRUTH_DIR / f"{page_id}.txt"
        reference = truth_path.read_text(encoding="utf-8").strip()
        ocr = str(row["ocr_text"])
        ref_norm = normalize_for_audit(reference)
        ocr_norm = normalize_for_audit(ocr)
        ref_hand = strip_form_labels(reference)
        ocr_hand = strip_form_labels(ocr)
        source_page = int(row["source_page_number"])
        rows.append(
            {
                "page_id": page_id,
                "source_page_number": source_page,
                "reference_length_raw": len(reference),
                "ocr_length_raw": len(ocr),
                "raw_levenshtein_distance": levenshtein(reference, ocr),
                "raw_character_cer": exact_cer(reference, ocr),
                "stored_cer": row["cer"],
                "stored_cer_matches_raw": math.isclose(
                    exact_cer(reference, ocr), float(row["cer"]), rel_tol=1e-12
                ),
                "whitespace_normalized_cer": exact_cer(ref_norm, ocr_norm),
                "handwritten_variant_cer_heuristic": exact_cer(ref_hand, ocr_hand),
                "reference_length_normalized": len(ref_norm),
                "ocr_length_normalized": len(ocr_norm),
                "ocr_confidence": row["ocr_confidence"],
                "reference_preview": ref_norm[:240],
                "ocr_preview": ocr_norm[:240],
                "discrepancy_excerpt": excerpt_diff(ref_norm, ocr_norm),
                "scope_note": (
                    "Heuristic form-label removal; not independently validated "
                    "and not used to replace author-verified CER."
                ),
            }
        )
        mapping_rows.append(
            {
                "page_id": page_id,
                "source_pdf": row["source_filename"],
                "source_page_number": source_page,
                "transcription_file": truth_path.name,
                "mapping_status": "matched_by_page_id_and_source_page",
                "ground_truth_status": row["ground_truth_status"],
            }
        )
    cer = pd.DataFrame(rows)
    mapping = pd.DataFrame(mapping_rows)
    cer.to_csv(AUDIT / "cer_audit.csv", index=False)
    mapping.to_csv(AUDIT / "ground_truth_mapping.csv", index=False)
    return cer, mapping


def fragmentation_features(image_path: Path) -> Tuple[dict, np.ndarray, np.ndarray]:
    cv2_mod = _ensure_cv2()
    _, binary, _ = preprocess_image(str(image_path))
    kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (5, 5))
    closed = cv2_mod.morphologyEx(
        binary, cv2_mod.MORPH_CLOSE, kernel, iterations=2
    )
    num_labels, labels, stats, centroids = cv2_mod.connectedComponentsWithStats(
        closed, connectivity=8
    )
    components = []
    for index in range(1, num_labels):
        area = int(stats[index, cv2_mod.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        components.append(
            {
                "area": area,
                "bbox": (
                    int(stats[index, cv2_mod.CC_STAT_LEFT]),
                    int(stats[index, cv2_mod.CC_STAT_TOP]),
                    int(stats[index, cv2_mod.CC_STAT_WIDTH]),
                    int(stats[index, cv2_mod.CC_STAT_HEIGHT]),
                ),
                "centroid": tuple(float(value) for value in centroids[index]),
            }
        )
    if len(components) <= 1:
        return {
            "raw_connected_component_count": num_labels - 1,
            "foreground_ink_area": sum(c["area"] for c in components),
            "component_count_after_min_area": len(components),
            "component_density_raw": None,
            "component_density_normalized": None,
            "mean_nearest_neighbor_distance": None,
            "nearest_neighbor_normalized": None,
            "small_component_count": None,
            "small_component_ratio": None,
            "weighted_raw_fragmentation_F": 0.0,
            "fragmentation_readability_f": 1.0,
        }, binary, closed
    ink_area = sum(c["area"] for c in components)
    density_raw = len(components) / max(ink_area, 1)
    centroids_array = np.array([c["centroid"] for c in components], dtype=np.float32)
    distances = []
    for index in range(len(centroids_array)):
        diff = centroids_array - centroids_array[index]
        distance = np.sqrt(np.sum(diff**2, axis=1))
        distance[index] = np.inf
        distances.append(float(np.min(distance)))
    avg_nn = float(np.mean(distances))
    page_h, page_w = closed.shape
    dispersion_raw = avg_nn / math.sqrt(float(page_h * page_w))
    density_norm = min(density_raw * 40, 1.0)
    dispersion_norm = min(dispersion_raw, 1.0)
    small_count = sum(c["area"] < SMALL_AREA for c in components)
    small_ratio = min(small_count / len(components), 1.0)
    fragmentation = (
        0.45 * density_norm + 0.35 * dispersion_norm + 0.20 * small_ratio
    )
    f = max(0.0, min(1.0, 1.0 - fragmentation))
    return {
        "raw_connected_component_count": num_labels - 1,
        "foreground_ink_area": ink_area,
        "component_count_after_min_area": len(components),
        "component_density_raw": density_raw,
        "component_density_normalized": density_norm,
        "mean_nearest_neighbor_distance": avg_nn,
        "nearest_neighbor_normalized": dispersion_norm,
        "small_component_count": small_count,
        "small_component_ratio": small_ratio,
        "weighted_raw_fragmentation_F": fragmentation,
        "fragmentation_readability_f": f,
    }, binary, closed


def audit_fragmentation() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    metrics = pd.read_csv(EXPERIMENT / "structural_metrics.csv")
    rows: List[dict] = []
    diagnostics = AUDIT / "visual_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for item in manifest.to_dict("records"):
        image_path = IMAGE_DIR / item["image_filename"]
        features, binary, closed = fragmentation_features(image_path)
        stored = metrics.loc[metrics["page_id"] == item["page_id"]].iloc[0]
        features.update(
            {
                "page_id": item["page_id"],
                "notebook_id": item["notebook_id"],
                "source_page_number": item["source_page_number"],
                "stored_text_fragmentation": stored["text_fragmentation"],
                "fragmentation_absolute_difference": abs(
                    features["fragmentation_readability_f"]
                    - float(stored["text_fragmentation"])
                ),
                "reproduces_stored_f": math.isclose(
                    features["fragmentation_readability_f"],
                    float(stored["text_fragmentation"]),
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                ),
            }
        )
        rows.append(features)
        if int(item["source_page_number"]) in DIAGNOSTIC_PAGES and item[
            "notebook_id"
        ] == "alyssa_bordner_i_1":
            page_dir = diagnostics / f"page_{item['source_page_number']:04d}"
            page_dir.mkdir(exist_ok=True)
            shutil.copy2(image_path, page_dir / "original.png")
            cv2.imwrite(str(page_dir / "binary.png"), binary)
            cv2.imwrite(str(page_dir / "closed.png"), closed)
            overlay = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            cv2_mod = _ensure_cv2()
            n, labels, stats, centroids = cv2_mod.connectedComponentsWithStats(
                closed, connectivity=8
            )
            for index in range(1, n):
                area = int(stats[index, cv2_mod.CC_STAT_AREA])
                if area < MIN_AREA:
                    continue
                x = int(stats[index, cv2_mod.CC_STAT_LEFT])
                y = int(stats[index, cv2_mod.CC_STAT_TOP])
                w = int(stats[index, cv2_mod.CC_STAT_WIDTH])
                h = int(stats[index, cv2_mod.CC_STAT_HEIGHT])
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cx, cy = centroids[index]
                cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 0, 0), -1)
            cv2.imwrite(str(page_dir / "components_bounding_boxes.png"), overlay)
    result = pd.DataFrame(rows)
    result.to_csv(AUDIT / "fragmentation_intermediate_metrics.csv", index=False)
    return result


def stats_report(cer: pd.DataFrame, frag: pd.DataFrame) -> pd.DataFrame:
    data = cer.merge(
        frag[
            [
                "page_id",
                "component_density_normalized",
                "nearest_neighbor_normalized",
                "small_component_ratio",
                "fragmentation_readability_f",
            ]
        ],
        on="page_id",
    )
    candidates = [
        ("component_density_normalized", "raw_character_cer"),
        ("nearest_neighbor_normalized", "raw_character_cer"),
        ("small_component_ratio", "raw_character_cer"),
        ("fragmentation_readability_f", "raw_character_cer"),
        ("component_density_normalized", "whitespace_normalized_cer"),
        ("nearest_neighbor_normalized", "whitespace_normalized_cer"),
        ("small_component_ratio", "whitespace_normalized_cer"),
        ("fragmentation_readability_f", "whitespace_normalized_cer"),
        ("component_density_normalized", "handwritten_variant_cer_heuristic"),
        ("nearest_neighbor_normalized", "handwritten_variant_cer_heuristic"),
        ("small_component_ratio", "handwritten_variant_cer_heuristic"),
        ("fragmentation_readability_f", "handwritten_variant_cer_heuristic"),
    ]
    rows: List[dict] = []
    p_values = []
    for metric, target in candidates:
        subset = data[[metric, target]].dropna()
        pearson = pearsonr(subset[metric], subset[target])
        spearman = spearmanr(subset[metric], subset[target])
        rows.extend(
            [
                {
                    "metric": metric,
                    "target_cer_variant": target,
                    "method": "pearson",
                    "n": len(subset),
                    "correlation": pearson.statistic,
                    "p_value": pearson.pvalue,
                },
                {
                    "metric": metric,
                    "target_cer_variant": target,
                    "method": "spearman",
                    "n": len(subset),
                    "correlation": spearman.statistic,
                    "p_value": spearman.pvalue,
                },
            ]
        )
        p_values.extend([pearson.pvalue, spearman.pvalue])
    adjusted = multipletests(p_values, method="bonferroni")[1]
    for row, adjusted_p in zip(rows, adjusted):
        row["bonferroni_p_value"] = adjusted_p
    output = pd.DataFrame(rows)
    output.to_csv(AUDIT / "component_correlations.csv", index=False)
    return output


def write_discrepancy_report(cer: pd.DataFrame) -> None:
    lines = [
        "# CER discrepancy report",
        "",
        "The source page mapping is verified by the page IDs and manifest: each "
        "Alyssa transcription file uses the corresponding source PDF page number.",
        "",
        "The stored CER uses the existing `compute_cer` path. This audit also "
        "computes exact character-level Levenshtein distance divided by raw "
        "reference length, plus whitespace-normalized and conservative "
        "form-label-filtered variants. The filtered variant is diagnostic only.",
        "",
        "The OCR was generated from the full rendered page. Therefore it can "
        "include printed labels, page numbers, signatures, dates, and form "
        "instructions that are not consistently represented in the handwritten "
        "transcriptions. The CSV contains previews and diff excerpts for manual review.",
        "",
        "Unicode is read as UTF-8. Raw CER preserves whitespace, punctuation, "
        "capitalization, and Unicode code points. The normalized variant collapses "
        "runs of whitespace only; it does not change case or punctuation. No "
        "ground-truth text was modified.",
        "",
        "The handwritten-content variant removes only literal form-label phrases "
        "such as Project, Continued from Page, Signed, Date, and Read and "
        "Understood By. It cannot reliably remove all printed text from flattened "
        "OCR, so it must not be presented as a validated full handwritten-only CER.",
        "",
        "Fragmentation diagnostics retain the existing preprocessing, area cutoff, "
        "normalization, and weights. Component bounding boxes may include chemical "
        "structures, table/grid lines, punctuation, and subscripts because the "
        "algorithm does not classify component semantics.",
    ]
    (AUDIT / "discrepancy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    cer, _ = audit_cer()
    frag = audit_fragmentation()
    correlations = stats_report(cer, frag)
    write_discrepancy_report(cer)
    summary = {
        "pages_audited": int(len(cer)),
        "raw_cer_matches_stored_count": int(cer["stored_cer_matches_raw"].sum()),
        "fragmentation_reproduction_count": int(frag["reproduces_stored_f"].sum()),
        "diagnostic_pages": sorted(DIAGNOSTIC_PAGES),
        "correlation_tests": int(len(correlations)),
        "audit_directory": str(AUDIT),
    }
    (AUDIT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
