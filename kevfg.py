import os
import json
from pathlib import Path
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
from doclayout_yolo import YOLOv10
import re
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Set paths
BASE_DIR = "pdf_files"
BASE_DIR2 = "model"

input_dir = os.path.join(BASE_DIR, "PDFS")
output_dir = os.path.join(BASE_DIR, "output")
output_image_dir = os.path.join(output_dir, "processed_images")
model_path = os.path.join(BASE_DIR2, "model.pt")


def extract_text_from_pdf(pdf_path):
    text_per_page = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_per_page.append(page.get_text())
        doc.close()
        print(f"Extracted text from {len(text_per_page)} pages of {pdf_path}")
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {e}")
        text_per_page = []
    return text_per_page


def convert_pdf_page_to_image(pdf_path, page_number, dpi=200):
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error converting page {page_number} of {pdf_path} to image: {e}")
        return None


def _is_title_like(name: str) -> bool:
    n = name.lower().replace("_", "-").strip()
    if "page" in n and "header" in n:
        return False
    if "title" in n:
        return True
    if "section" in n and "header" in n:
        return True
    if "heading" in n:
        return True
    return False


def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    t = re.sub(r"^[•\-\u2022\.\s]+", "", t)
    return t


def _cluster_levels_by_height(cands):
    if not cands:
        return []
    K = min(3, len(cands))
    heights = np.array([[c["h_px"]] for c in cands], dtype=float)

    if _HAS_SKLEARN and K >= 2:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(heights)
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)[::-1]
    else:
        if K == 1:
            labels = np.zeros(len(cands), dtype=int)
            order = [0]
        else:
            qs = np.quantile(heights.ravel(), np.linspace(0, 1, K + 1))
            labels = np.zeros(len(cands), dtype=int)
            for i, h in enumerate(heights.ravel()):
                for b in range(K):
                    if qs[b] <= h <= qs[b + 1]:
                        labels[i] = min(b, K - 1)
                        break
            means = [np.mean(heights.ravel()[labels == b]) for b in range(K)]
            order = np.argsort(means)[::-1]

    cluster_to_level = {order[i]: f"H{i+1}" for i in range(K)}
    return [cluster_to_level[int(lab)] for lab in labels]


def process_pdfs():
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    DPI = 200
    _SCALE = DPI / 72.0

    print("▶ Starting PDF layout extraction...")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        return

    print(f"[INFO] Loading DocLayout-YOLO model from {model_path}")
    model = YOLOv10(model_path)
    print("[INFO] Model loaded.")

    # Find every directory named "PDFS" under BASE_DIR (including nested)
    pdf_dirs = [p for p in Path(BASE_DIR).rglob("PDFS") if p.is_dir()]

    for pdf_dir in pdf_dirs:
        print(f"\n=== Scanning folder: {pdf_dir} ===")
        for pdf_path in pdf_dir.glob("*.pdf"):
            if pdf_path.name.startswith("._"):
                continue  # skip macOS ghost files
            pdf_file = pdf_path.name
            print(f"\n--- Processing PDF: {pdf_file} ---")

            extracted_text = extract_text_from_pdf(str(pdf_path))

            doc = fitz.open(str(pdf_path))
            title_candidates = []

            for page_num in range(doc.page_count):
                pil_image = convert_pdf_page_to_image(str(pdf_path), page_num, dpi=DPI)
                if pil_image is None:
                    continue

                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                temp_image_path = os.path.join(output_image_dir, f"temp_{pdf_file}_page_{page_num + 1}.jpg")
                cv2.imwrite(temp_image_path, opencv_image)

                try:
                    det_res = model.predict(temp_image_path, imgsz=1024, conf=0.2)
                    res = det_res[0]
                    # ---------- save annotated page image ----------
                    annotated = res.plot(pil=True, line_width=5, font_size=20)
                    result_image_path = os.path.join(
                        output_image_dir,
                        f"result_{pdf_path.stem}_page_{page_num + 1}.jpg"
                    )
                    cv2.imwrite(
                        result_image_path,
                        cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    )
                    # ------------------------------------------------

                    names = getattr(res, "names", {}) or {}
                    if not names and hasattr(res, "boxes") and hasattr(res.boxes, "cls"):
                        uniq = np.unique(res.boxes.cls.cpu().numpy()).astype(int).tolist()
                        names = {i: str(i) for i in uniq}

                    wanted_ids = {i for i, n in names.items() if _is_title_like(str(n))}

                    if not hasattr(res, "boxes") or res.boxes is None or not len(res.boxes):
                        continue

                    xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy().astype(float)

                    page_obj = doc[page_num]
                    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                        if cls_id not in wanted_ids:
                            continue

                        rect = fitz.Rect(x1/_SCALE, y1/_SCALE, x2/_SCALE, y2/_SCALE)
                        txt = _clean_text(page_obj.get_text("text", clip=rect))
                        if not txt:
                            continue

                        h_px = float(y2 - y1)
                        area_px = float((x2 - x1) * (y2 - y1))
                        y_center = float(0.5 * (y1 + y2))

                        title_candidates.append({
                            "page": page_num,
                            "text": txt,
                            "conf": float(conf),
                            "x1_px": float(x1), "y1_px": float(y1),
                            "x2_px": float(x2), "y2_px": float(y2),
                            "h_px": h_px,
                            "area_px": area_px,
                            "y_center_px": y_center,
                            "label_name": str(names.get(int(cls_id), cls_id)),
                        })

                except Exception as e:
                    print(f"Error on {pdf_file} page {page_num}: {e}")
                finally:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            doc.close()

            dedup = {}
            for c in title_candidates:
                key = (c["page"], c["text"].lower())
                if key not in dedup or c["conf"] > dedup[key]["conf"]:
                    dedup[key] = c
            candidates = list(dedup.values())

            doc_title = ""
            outline_items = []

            if candidates:
                title_cand = sorted(
                    candidates,
                    key=lambda c: (c["h_px"], c["conf"], -c["page"], -c["y1_px"]),
                    reverse=True
                )[0]
                doc_title = title_cand["text"]
                candidates = [c for c in candidates if not (
                    c["page"] == title_cand["page"] and c["text"] == title_cand["text"]
                )]

            if candidates:
                levels = _cluster_levels_by_height(candidates)
                for c, lvl in zip(candidates, levels):
                    c["level"] = lvl

                candidates.sort(key=lambda c: (c["page"], c["y1_px"]))

                outline_items = [
                    {"level": c["level"], "text": c["text"], "page": c["page"]}
                    for c in candidates
                ]

            out = {"title": doc_title, "outline": outline_items}
            out_path = os.path.join(
                output_dir,
                f"{pdf_path.stem}.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)

            print(f"✔ Saved outline for {pdf_file} → {out_path}")


if __name__ == "__main__":
    process_pdfs()
    print("Done.")