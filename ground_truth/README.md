# Ground truth transcriptions

Manual reference transcriptions used to calculate CER. The working copies the
code reads are one UTF-8 `.txt` file per page in
`experiments/expanded_dataset/ground_truth_provided/`, named
`<notebook_id>_p<PDF page, 4 digits>.txt`. Page numbers are PDF page numbers,
not the handwritten page numbers in the notebook.

| Notebook | Pages (PDF) | Source in this folder |
|---|---|---|
| `alyssa_bordner_i_1` | 23, 32, 43, 61, 63, 78, 79, 82, 85, 98 | `Alyssa Bordner I (10 transcribed pages).pdf`, split into per-page files by `src/prepare_expanded_dataset.py` |
| `wesley_newsome_1` | 5, 21, 27, 59, 64, 74, 75, 84, 88, 97 | `submitted_transcriptions/Wesley_pg*.txt` |
| `yuen_lau_notebook_1` | 10, 12, 23, 26, 45, 49, 64, 70, 72, 94 | `submitted_transcriptions/Yuen_pg*.txt` |

## Conventions

- Text is transcribed as written on the page, including printed form text
  where the transcriber included it.
- Tables are written as one line, read row by row and left to right, with
  cells separated by single spaces and empty cells omitted. Characters written
  in a cell (for example "-") are kept. The Wesley submissions used an XML
  table layout; the per-page files contain the same cells in this flattened
  form. The submitted originals are kept unchanged in
  `submitted_transcriptions/`.
- Crossed-out text is marked `~~like this~~` and uncertain readings
  `[?like this?]`. These markers are currently left in the files. A rule for
  handling them before CER is calculated has not yet been applied.
- Each per-page file was checked against the rendered PDF page to confirm the
  page number.
