# Expanded dataset preparation

This experiment preserves the original project outputs and creates a separate
30-page pilot dataset:

- 10 labeled pages from `Alyssa Bordner I-1.pdf`, matching the supplied
  transcription document.
- 10 evenly spaced pages from each of the Wesley and Yuen PDFs.

Pages are rendered from the source PDFs at 300 dpi into `images/`. The source
PDFs are not modified. Structural metrics are recorded in
`structural_metrics.csv`, and page provenance is recorded in
`dataset_manifest.csv`.

The complete combined results are in `page_level_results.csv`. Separate
per-notebook result tables are in `per_notebook/`:

- `alyssa_bordner_i_1_results.csv` — 10 pages, with author-verified CER.
- `wesley_newsome_1_results.csv` — 10 pages, OCR confidence only.
- `yuen_lau_notebook_1_results.csv` — 10 pages, OCR confidence only.

The source PDFs remain in the project-level `pdfs/` folder and are not copied
into this experiment package. The share ZIP contains the rendered sampled
pages and all result tables.

The supplied Alyssa transcription PDF was split into the ten labeled page
sections under `ground_truth_provided/`. Based on the project owner’s statement
that these were manually transcribed by the author, these entries are marked
`author_verified` in `transcription_status.csv`. This records author
verification, not an independent second review. All other selected pages are
marked `required` and need manual transcription.

Manual transcriptions for 10 pages each of the Wesley and Yuen notebooks were
added later in the same per-page format (see `ground_truth/README.md`). Only
Yuen pages 12 and 23 are part of the 30-page sample above; the other 18 pages
are outside it, and `dataset_manifest.csv`, `transcription_status.csv` and the
result tables have not been regenerated to include them.

The selection is a reproducible pilot sampling frame, not a claim that the
pages are independent observations or that visual categories have been
independently assigned. Complete the category, content, and verification
columns before the OCR/CER analysis.

## Reproduce the current run

From the project root:

```text
.venv\Scripts\python.exe src\prepare_expanded_dataset.py
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
.venv\Scripts\python.exe src\run_expanded_analysis.py
.venv\Scripts\python.exe src\make_expanded_figures.py
```

The analysis produces OCR confidence for all 30 pages, but CER only for the
10 Alyssa pages with author-verified transcriptions. It does not impute CER for
the other 20 pages.
