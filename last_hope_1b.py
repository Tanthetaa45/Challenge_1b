"""
Generate challenge1b_output.json for every Collection
using ONLY local Sentence-Transformers (no external API calls)
"""

import json, os, sys, re
from pathlib import Path
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ───────────────────────── CONFIG ──────────────────────────
ROOT                = Path("Challenge_1b")
LOCAL_SLM_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K               = 5                       # ← keep only the top 5
# ────────────────────────────────────────────────────────────

slm = SentenceTransformer(LOCAL_SLM_NAME, device="cpu")
print(f"[INFO] Sentence-Transformer «{LOCAL_SLM_NAME}» loaded.")

def cosine_scores(query: str, candidates: list[str]) -> list[float]:
    q = slm.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    c = slm.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(q, c).cpu().numpy().ravel().tolist()

# ---------------------------------------------------------------------
def load_collection_inputs(coll_dir: Path):
    """
    Reads persona & job_to_be_done from  challenge1b_input.json.
    Works with either of the two layouts shown above.
    """
    inp = coll_dir / "challenge1b_input.json"
    default_persona = "Travel Planner"
    default_job     = "Plan a trip of 4 days for a group of 10 college friends."

    if not inp.exists():
        return default_persona, default_job

    try:
        data = json.loads(inp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Cannot parse {inp.name}: {e}")
        return default_persona, default_job

    # layout A: whole object is under “metadata”
    if "metadata" in data and isinstance(data["metadata"], dict):
        meta = data["metadata"]
    else:               # layout B: keys are at top level
        meta = data

    return (
        meta.get("persona", default_persona),
        meta.get("job_to_be_done", default_job)
    )
# ---------------------------------------------------------------------


def gather_headings(out_json_dir: Path):
    headings, snippets, docs = [], [], []
    for f in sorted(out_json_dir.glob("*.json")):
        data = json.loads(f.read_text())
        doc = f.stem + ".pdf"
        docs.append(doc)
        for it in data.get("outline", []):
            headings.append({"document": doc,
                             "section_title": it["text"],
                             "page_number": it["page"] + 1})
            snippets.append({"document": doc,
                             "refined_text": it["text"][:200],
                             "page_number": it["page"] + 1})
    return docs, headings, snippets

def build_output(coll_dir: Path):
    persona, job = load_collection_inputs(coll_dir)
    out_dir = coll_dir / "output_json"
    if not out_dir.exists():
        print(f"[WARN] {out_dir} missing")
        return

    docs, heads, snips = gather_headings(out_dir)
    if not heads:
        print(f"[WARN] No heading JSONs in {out_dir}")
        return

    scores = cosine_scores(job, [h["section_title"] for h in heads])
    for h, s in zip(heads, scores):
        h["score"] = s
    heads.sort(key=lambda x: -x["score"])
    selected = heads[:TOP_K]

    for i, h in enumerate(selected, 1):
        h["importance_rank"] = i
        h.pop("score", None)

    sel_idx = {(h["document"], h["page_number"]) for h in selected}
    selected_snips = [s for s in snips if (s["document"], s["page_number"]) in sel_idx]

    payload = {
        "metadata": {
            "input_documents": docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": selected,
        "subsection_analysis": selected_snips
    }

    out_path = out_dir / "challenge1b_output.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[OK] {out_path.relative_to(ROOT.parent)}")

if __name__ == "__main__":
    if not ROOT.exists():
        sys.exit(f"No Challenge_1b directory at {ROOT.resolve()}")
    for coll in sorted(ROOT.iterdir()):
        if (coll / "output_json").is_dir():
            print(f"\n=== {coll.name} ===")
            build_output(coll)
    print("\nDone – final collection JSONs are ready.")
