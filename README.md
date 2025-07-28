# Adobe India Hackathon 2025 – Challenge 1B  
**Automated PDF-Outline Extraction & Top-5 Section Ranking**

### 📑 Project Purpose  
This repository delivers a two-stage, fully-offline pipeline that:

1. Detects headings in every page of the challenge PDFs with **DocLayout-YOLO**  
2. Ranks those headings against each collection’s *job-to-be-done* using a local **Sentence-Transformers** model, keeping **only the top 5** most-relevant sections  

Final JSONs (`challenge1b_output.json`) are produced per collection and are ready for submission to Adobe’s scorer.

---

## 1 Repository Layout


adobe_hackathon_1b/
├─ main/ # core modules (YOLO wrapper, helpers)
│ └─ doclayout_yolo.py
├─ models/
│ └─ model.pt # ← add YOLO weights here
├─ Challenge_1b/
│ └─ Collection n/
│ ├─ PDFs/ # input PDFs from Adobe
│ ├─ output_json/ # per-PDF outlines + final JSON
│ └─ processed_images/ # annotated page images
├─ extract_headings.py # step ① – heading detection
├─ rank_headings_top5.py # step ② – local SLM ranking
├─ requirements.txt
└─ README.md


---

## 2 Quick Start

### 2.1 Clone & set up


git clone <your-fork-url> adobe_hackathon_1b
cd adobe_hackathon_1b
python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt


### 2.2 Add model weights
Download **`model.pt`** supplied by Adobe and place it at:

adobe_hackathon_1b/models/model.pt


---

## 3 Run the Pipeline

### 3.1 Step ① — Extract headings (per-PDF outlines)


cd Challenge_1b
zip -r submission_1b.zip Collection\ */output_json/challenge1b_output.json


Upload `submission_1b.zip` to the hackathon portal.

---

## 4 Script Details

| Script | Purpose | Typical speed (CPU) |
|--------|---------|---------------------|
| `extract_headings.py` | PDF → images → YOLO detection → per-PDF outline JSON | 0.5–2 s per page |
| `rank_headings_top5.py` | Local Sentence-Transformer ranks headings, outputs final JSON (TOP 5) | < 1 s for 200 headings |

---

## 5 Troubleshooting

| Symptom | Likely cause & fix |
|---------|-------------------|
| `ModuleNotFoundError: doclayout_yolo` | Ensure `main/doclayout_yolo.py` exists and import path matches. |
| `FileNotFoundError: models/model.pt` | Place the YOLO checkpoint in `models/`. |
| `[WARN] No heading JSONs in …/output_json` when ranking | Run `extract_headings.py` first so per-PDF JSONs exist. |
| CUDA OOM | Edit scripts to load models on CPU (`device="cpu"`). |

---

## 6 `.gitignore` Highlights


