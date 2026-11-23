from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pymupdf

try:
    from src.document_readability import compute_metrics
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.document_readability import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "expanded_dataset"
IMAGE_DIR = EXPERIMENT / "images"
TRANSCRIPTION_DIR = EXPERIMENT / "ground_truth_provided"
PDF_DIR = ROOT / "pdfs"
TRANSCRIPTION_PDF = (
    ROOT / "ground_truth" / "Alyssa Bordner I (10 transcribed pages).pdf"
)
DPI = 300


def slug(value: str) -> str:
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value


def evenly_spaced_pages(page_count: int, count: int) -> List[int]:
    if count > page_count:
        raise ValueError("Cannot select more pages than the PDF contains")
    if count == 1:
        return [1]
    return [
        round(1 + index * (page_count - 1) / (count - 1))
        for index in range(count)
    ]


def read_transcriptions() -> Dict[int, str]:
    if not TRANSCRIPTION_PDF.exists():
        return {}
    document = pymupdf.open(TRANSCRIPTION_PDF)
    text = "\n".join(page.get_text() for page in document)
    document.close()
    matches = list(re.finditer(r"(?m)^PAGE\s+(\d+)\s*$", text))
    result: Dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[int(match.group(1))] = text[start:end].strip()
    return result


def notebook_files() -> Sequence[Path]:
    return sorted(PDF_DIR.glob("*.pdf"))


def selected_pages(path: Path, page_count: int, transcribed: Iterable[int]) -> List[int]:
    if slug(path.name) == "alyssa_bordner_i_1":
        return sorted(set(transcribed))
    return evenly_spaced_pages(page_count, 10)


def render_and_measure(
    path: Path,
    notebook_id: str,
    pages: Sequence[int],
    transcriptions: Dict[int, str],
) -> List[dict]:
    document = pymupdf.open(path)
    has_provided_transcription = notebook_id == "alyssa_bordner_i_1"
    rows: List[dict] = []
    for page_number in pages:
        page_id = f"{notebook_id}_p{page_number:04d}"
        image_path = IMAGE_DIR / f"{page_id}.png"
        pixmap = document[page_number - 1].get_pixmap(dpi=DPI, alpha=False)
        pixmap.save(str(image_path))
        metrics = compute_metrics(str(image_path)).to_dict()
        rows.append(
            {
                "page_id": page_id,
                "notebook_id": notebook_id,
                "source_filename": path.name,
                "source_page_number": page_number,
                "image_filename": image_path.name,
                "image_width": pixmap.width,
                "image_height": pixmap.height,
                "handwriting_category": "",
                "content_notes": "",
                "ground_truth_status": (
                    "author_verified"
                    if has_provided_transcription and page_number in transcriptions
                    else "required"
                ),
                **metrics,
            }
        )
    document.close()
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
    transcriptions = read_transcriptions()
    manifest_rows: List[dict] = []
    metric_rows: List[dict] = []

    for path in notebook_files():
        document = pymupdf.open(path)
        notebook_id = slug(path.name)
        pages = selected_pages(path, document.page_count, transcriptions)
        document.close()
        rows = render_and_measure(path, notebook_id, pages, transcriptions)
        manifest_rows.extend(rows)
        metric_rows.extend(rows)

    for page_number, text in transcriptions.items():
        output = TRANSCRIPTION_DIR / f"alyssa_bordner_i_1_p{page_number:04d}.txt"
        output.write_text(text + "\n", encoding="utf-8")

    manifest_fields = [
        "page_id",
        "notebook_id",
        "source_filename",
        "source_page_number",
        "image_filename",
        "image_width",
        "image_height",
        "handwriting_category",
        "content_notes",
        "ground_truth_status",
    ]
    metric_fields = manifest_fields + [
        "baseline_consistency",
        "spatial_organization",
        "text_fragmentation",
        "combined_score",
    ]
    write_csv(EXPERIMENT / "dataset_manifest.csv", manifest_rows, manifest_fields)
    write_csv(EXPERIMENT / "structural_metrics.csv", metric_rows, metric_fields)

    status_rows = [
        {
            "page_id": row["page_id"],
            "notebook_id": row["notebook_id"],
            "source_page_number": row["source_page_number"],
            "status": row["ground_truth_status"],
            "transcription_file": (
                f"ground_truth_provided/{row['page_id']}.txt"
                if row["ground_truth_status"] == "author_verified"
                else ""
            ),
            "verification_notes": "",
        }
        for row in manifest_rows
    ]
    write_csv(
        EXPERIMENT / "transcription_status.csv",
        status_rows,
        [
            "page_id",
            "notebook_id",
            "source_page_number",
            "status",
            "transcription_file",
            "verification_notes",
        ],
    )


if __name__ == "__main__":
    main()
