import io
import os
import logging
from typing import Dict, Any, List

import numpy as np
import torch
import torch.serialization
from PIL import Image, ImageFile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Allow loading slightly broken images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-api")

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "cls_best.pt")

# IMPORTANT: match your training folder names ORDER (alphabetical for YOLOv5-cls)
CLASSES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = {"mel", "bcc", "akiec"}

IMG_SIZE = 224

# Step 8: decision threshold + risk bands
THRESHOLD = 0.45
HIGH_RISK = 0.75

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="Skin Lesion Classification API (YOLOv5-cls)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon mode
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# -----------------------------
# LOAD MODEL (PyTorch 2.6+ safe + YOLOv5 checkpoint)
# -----------------------------
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Missing model weights at: {WEIGHTS_PATH}")

# YOLOv5 stores a custom class in checkpoint, so we must allowlist it in PyTorch 2.6+
# This import requires yolov5/ to be on PYTHONPATH.
from models.yolo import ClassificationModel  # noqa: E402

torch.serialization.add_safe_globals([ClassificationModel])

ckpt = torch.load(
    WEIGHTS_PATH,
    map_location=device,
    weights_only=False,  # MUST for YOLOv5 checkpoints
)

model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model = model.to(device).eval()
logger.info("Model loaded OK")


def model_dtype(m: torch.nn.Module) -> torch.dtype:
    try:
        return next(m.parameters()).dtype
    except StopIteration:
        return torch.float32


MDTYPE = model_dtype(model)
logger.info(f"Model dtype: {MDTYPE}")

# CPU + fp16 can be flaky. Force float32 on CPU.
if device == "cpu" and MDTYPE == torch.float16:
    model = model.float()
    MDTYPE = torch.float32
    logger.info("Forced model to float32 on CPU")

# -----------------------------
# PREPROCESS (ImageNet norm)
# -----------------------------
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    x = torch.from_numpy(arr).unsqueeze(0)           # 1,3,H,W

    # Move first, then normalize on same device/dtype
    x = x.to(device=device, dtype=MDTYPE)
    mean = MEAN.to(device=device, dtype=MDTYPE)
    std = STD.to(device=device, dtype=MDTYPE)

    x = (x - mean) / std
    return x

# -----------------------------
# MULTI-CROP (Step 6)
# -----------------------------
def get_crops(img: Image.Image, crop_ratio: float = 0.80) -> List[Image.Image]:
    """
    Returns: [full_image, center, top-left, top-right, bottom-left, bottom-right]
    We crop a square (crop_ratio * min_side) from multiple positions.
    """
    img = img.convert("RGB")
    w, h = img.size
    min_side = min(w, h)

    crop_size = int(min_side * crop_ratio)
    crop_size = max(32, crop_size)  # safety

    crops: List[Image.Image] = [img]

    left = (w - crop_size) // 2
    top = (h - crop_size) // 2

    boxes = [
        (left, top),                    # center
        (0, 0),                         # top-left
        (w - crop_size, 0),             # top-right
        (0, h - crop_size),             # bottom-left
        (w - crop_size, h - crop_size), # bottom-right
    ]

    for x, y in boxes:
        x = max(0, min(x, w - crop_size))
        y = max(0, min(y, h - crop_size))
        crops.append(img.crop((x, y, x + crop_size, y + crop_size)))

    return crops


@torch.no_grad()
def predict_probs_single(img: Image.Image) -> torch.Tensor:
    x = preprocess_image(img)
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.softmax(logits, dim=1).squeeze(0)  # (C,)
    return probs


@torch.no_grad()
def predict_probs_multicrop(img: Image.Image, crop_ratio: float = 0.80) -> torch.Tensor:
    crops = get_crops(img, crop_ratio=crop_ratio)
    probs_list: List[torch.Tensor] = [predict_probs_single(c) for c in crops]
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (C,)
    return avg_probs

# -----------------------------
# INFERENCE
# -----------------------------
@torch.no_grad()
def run_inference(img: Image.Image) -> Dict[str, Any]:
    probs = predict_probs_multicrop(img, crop_ratio=0.80)

    if probs.numel() != len(CLASSES):
        return {
            "error": "Model class count mismatch",
            "model_outputs": int(probs.numel()),
            "expected_classes": int(len(CLASSES)),
            "expected_class_names": CLASSES,
        }

    probs_np = probs.detach().cpu().numpy()
    prob_map = {CLASSES[i]: float(probs_np[i]) for i in range(len(CLASSES))}

    top_idx = int(np.argmax(probs_np))
    top_class = CLASSES[top_idx]
    top_conf = float(probs_np[top_idx])

    malignancy_score = float(sum(prob_map[c] for c in MALIGNANT))
    benign_score = float(1.0 - malignancy_score)

    decision = "malignant" if malignancy_score >= THRESHOLD else "benign"

    if malignancy_score >= HIGH_RISK:
        risk_level = "high"
    elif malignancy_score >= THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "top_class": top_class,
        "confidence": top_conf,

        # binary outputs
        "decision": decision,
        "threshold": THRESHOLD,
        "risk_level": risk_level,

        # keep this for compatibility
        "cancerous": top_class in MALIGNANT,

        # scores
        "malignancy_score": malignancy_score,
        "benign_score": benign_score,

        # full class distribution
        "probabilities": prob_map,

        # debug/meta
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
        "strategy": "multicrop_avg",
        "note": "Scores are model confidence, not clinical probability.",
    }


@torch.no_grad()
def run_inference_debug(img: Image.Image) -> Dict[str, Any]:
    crop_ratio = 0.80
    crops = get_crops(img, crop_ratio=crop_ratio)

    per_crop: List[Dict[str, Any]] = []
    probs_list: List[torch.Tensor] = []

    for i, c in enumerate(crops):
        probs = predict_probs_single(c)  # (C,)
        if probs.numel() != len(CLASSES):
            return {
                "error": "Model class count mismatch (per-crop)",
                "crop_index": i,
                "model_outputs": int(probs.numel()),
                "expected_classes": int(len(CLASSES)),
                "expected_class_names": CLASSES,
            }

        probs_np = probs.detach().cpu().numpy()
        top_idx = int(np.argmax(probs_np))
        top_class = CLASSES[top_idx]
        top_conf = float(probs_np[top_idx])

        per_crop.append({
            "crop_index": i,
            "top_class": top_class,
            "confidence": top_conf,
        })
        probs_list.append(probs)

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (C,)
    avg_np = avg_probs.detach().cpu().numpy()
    prob_map = {CLASSES[i]: float(avg_np[i]) for i in range(len(CLASSES))}

    top_idx = int(np.argmax(avg_np))
    top_class = CLASSES[top_idx]
    top_conf = float(avg_np[top_idx])

    malignancy_score = float(sum(prob_map[c] for c in MALIGNANT))
    benign_score = float(1.0 - malignancy_score)

    decision = "malignant" if malignancy_score >= THRESHOLD else "benign"

    if malignancy_score >= HIGH_RISK:
        risk_level = "high"
    elif malignancy_score >= THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "final": {
            "top_class": top_class,
            "confidence": top_conf,
            "decision": decision,
            "threshold": THRESHOLD,
            "risk_level": risk_level,
            "malignancy_score": malignancy_score,
            "benign_score": benign_score,
            "probabilities": prob_map,
        },
        "per_crop": per_crop,
        "meta": {
            "crops_used": len(crops),
            "crop_ratio": crop_ratio,
            "device": device,
            "model_dtype": str(MDTYPE).replace("torch.", ""),
        },
        "note": "Debug endpoint shows per-crop predictions + averaged result.",
    }

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
        "classes": CLASSES,
        "malignant_classes": sorted(list(MALIGNANT)),
        "strategy": "multicrop_avg",
        "threshold": THRESHOLD,
        "high_risk_cutoff": HIGH_RISK,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        if not img_bytes:
            return {"error": "Empty file"}

        img = Image.open(io.BytesIO(img_bytes))
        return run_inference(img)

    except Exception as e:
        logger.exception("Analyze failed")
        return {"error": "Analyze failed", "detail": str(e)}


@app.post("/analyze_debug")
async def analyze_debug(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        if not img_bytes:
            return {"error": "Empty file"}

        img = Image.open(io.BytesIO(img_bytes))
        return run_inference_debug(img)

    except Exception as e:
        logger.exception("Analyze_debug failed")
        return {"error": "Analyze_debug failed", "detail": str(e)}
