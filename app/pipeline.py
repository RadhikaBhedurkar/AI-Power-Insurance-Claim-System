import os, re, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from PIL import Image
from pypdf import PdfReader
import pytesseract
from torchvision import models, transforms as T
from ultralytics import YOLO

SEV = ["minor", "moderate", "severe"]
CATS = ["collision", "parking damage", "glass damage", "theft or vandalism", "other"]
_cache = {}

def _yolo():
    if "y" not in _cache:
        _cache["y"] = YOLO(os.getenv("YOLO_WEIGHTS", "models/yolo_damage.pt"))
    return _cache["y"]

def _sev():
    if "s" not in _cache:
        m = models.resnet18(); m.fc = nn.Linear(512, 3)
        m.load_state_dict(torch.load("models/severity.pt", map_location="cpu")); _cache["s"] = m.eval()
    return _cache["s"]

def _zs():
    if "z" not in _cache:
        from transformers import pipeline
        _cache["z"] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _cache["z"]

_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def extract_document_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        return text if len(text.strip()) > 30 else ""  # scanned PDF: rasterize with pdf2image + OCR
    return pytesseract.image_to_string(Image.open(path))

def extract_entities(text: str) -> dict:
    pat = {"policy_number": r"policy\s*(?:no|number|#)?[:\s-]*([A-Z0-9/-]{6,})",
           "vehicle_reg": r"\b([A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4})\b",
           "date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
           "claimant": r"(?:name|insured)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)"}
    out = {}
    for k, v in pat.items():
        m = re.search(v, text, re.I if k == "policy_number" else 0)
        out[k] = m.group(1) if m else None
    out["missing_fields"] = [k for k, v in out.items() if v is None]
    return out

def classify_accident(desc: str) -> dict:
    if not desc.strip(): return {"category": "unknown", "confidence": 0.0}
    r = _zs()(desc, CATS)
    return {"category": r["labels"][0], "confidence": round(r["scores"][0], 3)}

def analyze_images(paths: list) -> dict:
    dets, probs = [], []
    for p in paths:
        for r in _yolo()(p, conf=0.25, verbose=False):
            for b in r.boxes:
                dets.append({"part": r.names[int(b.cls)], "conf": round(float(b.conf), 3), "box": [round(v) for v in b.xyxy[0].tolist()]})
        with torch.no_grad():
            probs.append(torch.softmax(_sev()(_tf(Image.open(p).convert("RGB"))[None]), 1)[0].numpy())
    avg = np.mean(probs, 0) if probs else np.zeros(3)
    return {"detections": dets, "severity": SEV[int(avg.argmax())], "severity_conf": round(float(avg.max()), 3), "severity_idx": int(avg.argmax())}

def estimate_cost(sev_idx: int, dets: list, vehicle_age: int = 5) -> list:
    lo, hi, cols = joblib.load("models/cost.joblib")
    names = " ".join(d["part"].lower() for d in dets)
    row = pd.DataFrame([{"severity": sev_idx, "n_parts": max(1, len(dets)), "glass": int("glass" in names),
                         "lamp": int("lamp" in names), "vehicle_age": vehicle_age}])[cols]
    a, b = float(lo.predict(row)[0]), float(hi.predict(row)[0])
    return [int(round(min(a, b), -3)), int(round(max(a, b), -3))]

def consistency_flags(acc, img, ent, desc) -> list:
    """Rule-based fusion. Upgrade path: a vision-language model comparing image + text."""
    f = []
    if acc["category"] == "parking damage" and img["severity"] == "severe": f.append("Described as parking damage but images show severe damage")
    if acc["category"] == "glass damage" and not any("glass" in x["part"].lower() for x in img["detections"]): f.append("Glass damage claimed but none detected")
    if not img["detections"]: f.append("No damage detected: images may be unclear or unrelated")
    if ent["missing_fields"]: f.append("Missing document fields: " + ", ".join(ent["missing_fields"]))
    return f

def assess(image_paths, doc_paths, description, vehicle_age=5) -> dict:
    text = "\n".join(extract_document_text(p) for p in doc_paths)
    ent = extract_entities(text)
    acc = classify_accident(description)
    img = analyze_images(image_paths)
    cost = estimate_cost(img["severity_idx"], img["detections"], vehicle_age)
    return {"entities": ent, "accident": acc, "damage": img, "cost_range_inr": cost,
            "flags": consistency_flags(acc, img, ent, description), "ocr_text": text[:3000]}
