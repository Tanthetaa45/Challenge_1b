#!/usr/bin/env python
"""
Create <CollectionName>_challenge1b_output.json for every collection.

Workflow:
  1. Assume each PDF already has an outline JSON created by the YOLO script:
         pdf_files/output/<PDF-stem>.json
  2. For every Collection (folder that contains challenge1b_input.json)
     rank headings against the persona's “job_to_be_done”
     and emit one summary JSON in /all_outputs/.
"""

import json, sys
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import fitz  # PyMuPDF

# ───────────── CONFIG ──────────────
ROOT            = Path("/Volumes/Extreme SSD/Challenge_1b")
COLLECTION_ROOT = ROOT / "pdf_files"                # where Collection X live
OUTLINE_ROOT    = ROOT / "pdf_files" / "output"     # where YOLO outlines live
OUTPUT_ROOT     = ROOT / "all_outputs"              # where summary JSONs go
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 5
RECALL_TOP_N         = 20                                 # candidates kept after bi‑encoder stage
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-12-v2"
SNIPPET_CHARS   = 1000
# ────────────────────────────────────

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

slm = SentenceTransformer(MODEL_NAME, device="cpu")
ce = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
print(f"[INFO] Sentence-Transformer loaded: {MODEL_NAME}")
print(f"[INFO] Cross-Encoder loaded: {CROSS_ENCODER_MODEL}")

# ---------- helpers -------------------------------------------------- #
def cosine_scores(query: str, candidates: list[str]) -> list[float]:
    q = slm.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    c = slm.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(q, c).cpu().numpy().ravel().tolist()


def load_outline(pdf_stem: str):
    """Return list of {'section_title', 'page_number'} from per-PDF outline."""
    fp = OUTLINE_ROOT / f"{pdf_stem}.json"
    if not fp.exists():
        return []          # caller decides what to do on miss
    data = json.loads(fp.read_text(encoding="utf-8"))
    return [
        {
            "section_title": item["text"],
            "page_number":  item["page"]
        }
        for item in data.get("outline", [])
    ]


def load_collection_inputs(coll_dir: Path):
    cfg = coll_dir / "challenge1b_input.json"
    if not cfg.exists():
        print(f"[SKIP] No challenge1b_input.json in {coll_dir}")
        return None, None, None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        docs    = [d["filename"] for d in data["documents"]]
        persona = data["persona"]["role"]
        job     = data["job_to_be_done"]["task"]
        return persona, job, docs
    except (KeyError, json.JSONDecodeError) as e:
        print(f"[WARN] Malformed {cfg}: {e}")
        return None, None, None



def safe_page_text(pdf_path: Path, page_idx: int) -> str:
    """Return text of page_idx (0-based) or '' on any failure."""
    try:
        with fitz.open(pdf_path) as doc:
            if 0 <= page_idx < doc.page_count:
                return doc[page_idx].get_text().replace("\n", " ")
    except Exception:
        pass
    return ""


# New helper function
def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """Return list of page texts, [] on any failure."""
    try:
        with fitz.open(pdf_path) as doc:
            return [p.get_text().replace("\n", " ") for p in doc]
    except Exception:
        return []


def gather_headings(coll_dir: Path, filenames: list[str]):
    """Return (docs, headings, snippets) ready for scoring (uses outlines)."""
    headings, snippets, docs = [], [], []
    pdf_folder = coll_dir / "PDFS"

    for fname in filenames:
        if fname.startswith("._"):
            continue
        pdf_stem = Path(fname).stem
        outline_items = load_outline(pdf_stem)

        # if outline missing, fall back to direct PDF scan
        if not outline_items:
            print(f"[MISS] outline for {fname}; opening PDF directly")
            outline_items = []
            pdf_path = pdf_folder / fname
            if not pdf_path.exists():
                print(f"[MISS] {fname} not found in {pdf_folder}")
                continue
            pages = extract_text_from_pdf(pdf_path)
            for idx, txt in enumerate(pages, 1):
                heading = txt.strip()[:256] if txt.strip() else ""
                outline_items.append({"section_title": heading, "page_number": idx})

        # record headings
        for it in outline_items:
            headings.append({
                "document": fname,
                "section_title": it["section_title"],
                "page_number":  it["page_number"]
            })

        docs.append(fname)

    return docs, headings


# --------------------------------------------------------------------- #
def build_output_for_collection(coll_dir: Path):
    persona, job, filenames = load_collection_inputs(coll_dir)
    if not (persona and job and filenames):
        return

    print(f"\n[PROCESS] {coll_dir.relative_to(ROOT)}")
    docs, headings = gather_headings(coll_dir, filenames)
    if not headings:
        print("        no headings found – skipped")
        return

    scores = cosine_scores(job, [h["section_title"] for h in headings])
    for h, s in zip(headings, scores):
        h["score"] = s

    # ── Stage‑1 (bi‑encoder) recall ────────────────────────────────
    headings.sort(key=lambda x: -x["score"])
    recall = headings[:RECALL_TOP_N]

    # ── Stage‑2 (cross‑encoder) rerank ────────────────────────────
    ce_pairs   = [(job, h["section_title"]) for h in recall]
    ce_scores  = ce.predict(ce_pairs)
    for h, s in zip(recall, ce_scores):
        h["ce_score"] = float(s)

    recall.sort(key=lambda x: -x["ce_score"])
    # Enforce diversity: pick at most one section per document
    top = []
    seen_docs = set()
    for h in recall:
        doc = h["document"]
        if doc not in seen_docs:
            top.append(h)
            seen_docs.add(doc)
        if len(top) == TOP_K:
            break

    # Finalise metadata for payload
    for rank, h in enumerate(top, 1):
        h["importance_rank"] = rank
        h.pop("score", None)
        h.pop("ce_score", None)

    # build subsection_analysis snippets **only for the TOP_K pages**
    snippets = []
    pdf_folder = coll_dir / "PDFS"
    for h in top:
        pdf_path = pdf_folder / h["document"]
        page_text = safe_page_text(pdf_path, h["page_number"] - 1)
        snippets.append({
            "document":     h["document"],
            "refined_text": page_text[:SNIPPET_CHARS],
            "page_number":  h["page_number"]
        })

    payload = {
        "metadata": {
            "input_documents": docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": top,
        "subsection_analysis": snippets
    }

    out_file = OUTPUT_ROOT / f"{coll_dir.name}_challenge1b_output.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"        → saved {out_file.name}")


# ================================ main ================================= #
if __name__ == "__main__":
    if not ROOT.exists():
        sys.exit(f"[FATAL] ROOT not found: {ROOT}")

    for coll in COLLECTION_ROOT.rglob("*"):
        if coll.is_dir() and (coll / "challenge1b_input.json").exists():
            build_output_for_collection(coll)

    print("\n[OK] All JSONs are in:", OUTPUT_ROOT)