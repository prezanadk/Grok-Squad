import io
import os
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "cls_best.pt")

# YOLOv5-cls folder order is alphabetical by class folder name:
# akiec, bcc, bkl, df, mel, nv, vasc
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = {"mel", "bcc", "akiec"}

IMG_SIZE = 224

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Skin Lesion Classification API (YOLOv5-cls)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon mode
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD MODEL (robust)
# =====================================================
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing model weights at: {WEIGHTS_PATH}")

ckpt = torch.load(WEIGHTS_PATH, map_location=device)

if isinstance(ckpt, dict) and "model" in ckpt:
    model = ckpt["model"]
else:
    model = ckpt

model = model.to(device).eval()

def model_dtype(m: torch.nn.Module) -> torch.dtype:
    """Get dtype of model parameters. Fallback float32 if unknown."""
    try:
        return next(m.parameters()).dtype
    except StopIteration:
        return torch.float32

MDTYPE = model_dtype(model)

# If you're on CPU and model is half, CPU half can be flaky.
# Force float32 on CPU to avoid weirdness.
if device == "cpu" and MDTYPE == torch.float16:
    model = model.float()
    MDTYPE = torch.float32

# =====================================================
# PREPROCESS
# =====================================================
def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, 0..1
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    x = torch.from_numpy(arr).unsqueeze(0)           # BCHW

    # ImageNet normalization (YOLOv5-cls standard)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std

    # Move + MATCH dtype to model weights (this fixes your error)
    x = x.to(device=device, dtype=MDTYPE)

    return x

# =====================================================
# INFERENCE
# =====================================================
@torch.no_grad()
def run_inference(img: Image.Image) -> Dict[str, Any]:
    x = preprocess_image(img)

    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probs = torch.softmax(logits, dim=1).squeeze(0)

    prob_map = {CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))}

    top_idx = int(torch.argmax(probs).item())
    top_class = CLASSES[top_idx]
    top_conf = float(probs[top_idx].item())

    malignancy_score = sum(prob_map[c] for c in MALIGNANT if c in prob_map)

    return {
        "top_class": top_class,
        "confidence": top_conf,
        "cancerous": top_class in MALIGNANT,
        "malignancy_score": malignancy_score,
        "probabilities": prob_map,
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
    }

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
        "classes": CLASSES,
        "malignant_classes": sorted(list(MALIGNANT)),
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        return run_inference(img)
    except Exception as e:
        return {"error": "Analyze failed", "detail": str(e)}
