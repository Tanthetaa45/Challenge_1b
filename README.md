# Adobe India Hackathon 2025 – Challenge 1B  
**Automated PDF‑Outline Extraction & Top‑5 Section Ranking (fully offline)**

## Project Purpose
This repository implements a two‑stage pipeline that runs fully on your machine:

1. **Heading extraction** for every PDF page (DocLayout‑YOLO or your extractor).
2. **Semantic ranking** of those headings against each collection’s *job‑to‑be‑done* using local Sentence‑Transformers (bi‑encoder recall + cross‑encoder rerank).

The pipeline emits one final JSON per collection:  
`all_outputs/<CollectionName>_challenge1b_output.json`.

---

## Repository Layout

```
Challenge_1b/
├─ all_outputs/                  # final per‑collection JSONs
├─ model/                        # optional weights or models
├─ output/                       # legacy outlines (older scripts)
├─ output_json/                  # legacy outlines (older scripts)
├─ pdf_files/
│  ├─ Collection 1/
│  │  ├─ PDFS/                   # input PDFs for this collection
│  │  └─ challenge1b_input.json  # persona + job + doc list
│  ├─ Collection 2/
│  └─ Collection 3/
├─ processed_images/             # annotated pages (if produced)
├─ Persona_pdfs.py               # Stage 2: rank & produce final JSONs
├─ process_pdfs.py               # Stage 1: your PDF → outline generator
├─ requirements.txt
└─ README.md
```

> **Note**  
> `Persona_pdfs.py` expects outline JSONs at `pdf_files/output/<PDF‑stem>.json`.  
> If your extractor writes elsewhere (e.g., repo‑root `output/`), either move the files or change `OUTLINE_ROOT` in `Persona_pdfs.py`.

---

## Quick Start

### 1) Environment
```bash
python3 -m venv .venv

# macOS (zsh/bash)
source .venv/bin/activate
# macOS (fish shell)
source .venv/bin/activate.fish
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Put PDFs and inputs in place
```
pdf_files/Collection X/PDFS/Your.pdf
pdf_files/Collection X/challenge1b_input.json
```

Minimal `challenge1b_input.json` example:
```json
{
  "documents": [
    {"filename": "South of France - Cities.pdf"},
    {"filename": "South of France - Cuisine.pdf"},
    {"filename": "South of France - History.pdf"},
    {"filename": "South of France - Restaurants and Hotels.pdf"},
    {"filename": "South of France - Things to Do.pdf"},
    {"filename": "South of France - Tips and Tricks.pdf"},
    {"filename": "South of France - Traditions and Culture.pdf"}
  ],
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a trip of 4 days for a group of 10 college friends." }
}
```

### 3) Stage 1 — Generate per‑PDF outlines
Run your extractor so that each `PDFS/*.pdf` has a matching outline JSON at:
```
pdf_files/output/<PDF‑stem>.json
```

### 4) Stage 2 — Rank & produce final JSONs
From repo root:
```bash
python Persona_pdfs.py
```
You should see logs like:
```
[PROCESS] pdf_files/Collection 1
        → saved Collection 1_challenge1b_output.json
```
Final files appear in:
```
all_outputs/<CollectionName>_challenge1b_output.json
```

### 5) Package for submission (optional)
```bash
cd all_outputs
zip -r submission_1b.zip *.json
```

---

## How it Works (short)
- **Recall (bi‑encoder)**: `all‑MiniLM‑L6‑v2` embeds the job text and candidate section titles, keeping the best `RECALL_TOP_N`.
- **Rerank (cross‑encoder)**: `ms‑marco‑MiniLM‑L‑12‑v2` scores `(job, title)` pairs jointly for sharper ordering.
- **Top‑K**: The best `TOP_K` become `extracted_sections`. For each, the script grabs page text (PyMuPDF) to build `subsection_analysis`.

Key tunables in `Persona_pdfs.py`:
```python
TOP_K = 5                    # final sections
RECALL_TOP_N = 20            # candidates sent to rerank
SNIPPET_CHARS = 1000         # snippet length
OUTLINE_ROOT = ROOT / "pdf_files" / "output"
```

---

## Troubleshooting

- **No final JSONs**: Ensure each collection has `challenge1b_input.json` and PDFs in `PDFS/`.
- **“no headings found – skipped”**: Missing outline JSONs. Confirm files exist under `pdf_files/output/` with names matching the PDF stems.
- **Wrong page numbers / snippets**: Outline JSONs must use **1‑based** page numbers (the script subtracts 1 only when extracting text).
- **Everything comes from one PDF**: Keep the cross‑encoder rerank enabled and make sure outlines exist for all PDFs (Cities, Things to Do, Cuisine, etc.).
- **Outlines landed in `output/` at repo root**: Move them to `pdf_files/output/` or update `OUTLINE_ROOT`.

---

## Requirements
- Python 3.9+
- `sentence-transformers`, `torch`, `transformers`, `PyMuPDF` (fitz)

Install all with:
```bash
pip install -r requirements.txt
```

> Runs on CPU by default. To enable CUDA, change the `device` for both models in `Persona_pdfs.py`.

---

## .gitignore tips (macOS)
Add these if you see Finder sidecar files in `git status`:
```
.DS_Store
._*
.AppleDouble
.Spotlight-V100
.Trashes
.fseventsd
Icon?
```

---

## License
Internal use for Adobe India Hackathon 2025 – Challenge 1B. Adapt as needed for your submission.


