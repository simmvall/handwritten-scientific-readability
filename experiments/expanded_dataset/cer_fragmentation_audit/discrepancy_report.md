# CER discrepancy report

The source page mapping is verified by the page IDs and manifest: each Alyssa transcription file uses the corresponding source PDF page number.

The stored CER uses the existing `compute_cer` path. This audit also computes exact character-level Levenshtein distance divided by raw reference length, plus whitespace-normalized and conservative form-label-filtered variants. The filtered variant is diagnostic only.

The OCR was generated from the full rendered page. Therefore it can include printed labels, page numbers, signatures, dates, and form instructions that are not consistently represented in the handwritten transcriptions. The CSV contains previews and diff excerpts for manual review.

Unicode is read as UTF-8. Raw CER preserves whitespace, punctuation, capitalization, and Unicode code points. The normalized variant collapses runs of whitespace only; it does not change case or punctuation. No ground-truth text was modified.

The handwritten-content variant removes only literal form-label phrases such as Project, Continued from Page, Signed, Date, and Read and Understood By. It cannot reliably remove all printed text from flattened OCR, so it must not be presented as a validated full handwritten-only CER.

Fragmentation diagnostics retain the existing preprocessing, area cutoff, normalization, and weights. Component bounding boxes may include chemical structures, table/grid lines, punctuation, and subscripts because the algorithm does not classify component semantics.
