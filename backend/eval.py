import io
import os
import logging
from typing import Dict, Any, List, Optional, Tuple

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

# MUST match your training folder names (YOLOv5-cls uses alphabetical folder order)
CLASSES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = {"mel", "bcc", "akiec"}  # treat these as malignant for binary decision

IMG_SIZE = 224

# Decision thresholds (tune these)
THRESHOLD = 0.45          # malignant if malignancy_score >= this
UNCERTAIN_LOW = 0.40      # uncertain band
UNCERTAIN_HIGH = 0.55
HIGH_RISK = 0.75          # risk label cutoff

# Multicrop config
CROP_RATIO = 0.80

# Image quality gates (external images)
MIN_SIDE_PX = 96          # reject tiny images
DARK_MEAN_MIN = 25        # on 0-255 grayscale mean
BLUR_VAR_MIN = 50.0       # Laplacian variance threshold (lower = blurrier)

# -----------------------------
# FastAPI app & CORS
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

# IMPORTANT:
# For YOLOv5 checkpoints on PyTorch 2.6+, allowlist the YOLOv5 class.
# This requires that yolov5 is on PYTHONPATH when you run uvicorn, e.g.:
#   set PYTHONPATH=%cd%\yolov5  (from project root)
# or run uvicorn from root with PYTHONPATH set.
from models.yolo import ClassificationModel  # noqa: E402

torch.serialization.add_safe_globals([ClassificationModel])

ckpt = torch.load(
    WEIGHTS_PATH,
    map_location=device,
    weights_only=False,  # required for YOLOv5 checkpoints that include custom classes
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
# QUALITY CHECKS (for external photos)
# -----------------------------
def _to_gray_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def image_quality_check(img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Returns None if OK.
    Returns an error dict if image is too small, too dark, or too blurry.
    """
    w, h = img.size
    if min(w, h) < MIN_SIDE_PX:
        return {
            "error": "Image too small",
            "detail": f"Minimum side must be >= {MIN_SIDE_PX}px, got {w}x{h}.",
        }

    gray = _to_gray_np(img)
    mean_intensity = float(gray.mean())
    if mean_intensity < DARK_MEAN_MIN:
        return {
            "error": "Image too dark",
            "detail": f"Mean brightness {mean_intensity:.1f} < {DARK_MEAN_MIN}. Use better lighting.",
        }

    # Simple blur estimate using Laplacian variance (no OpenCV needed)
    # Laplacian kernel:
    #   0  1  0
    #   1 -4  1
    #   0  1  0
    g = gray.astype(np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
        + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
    )
    blur_var = float(lap.var())
    if blur_var < BLUR_VAR_MIN:
        return {
            "error": "Image too blurry",
            "detail": f"Blur score {blur_var:.1f} < {BLUR_VAR_MIN}. Hold steady and focus closer.",
        }

    return None


# -----------------------------
# MULTI-CROP
# -----------------------------
def get_crops(img: Image.Image, crop_ratio: float = 0.80) -> List[Image.Image]:
    """
    Returns crops: [full, center, top-left, top-right, bottom-left, bottom-right]
    Crop size = crop_ratio * min_side (square crop).
    """
    img = img.convert("RGB")
    w, h = img.size
    min_side = min(w, h)

    crop_size = max(32, int(min_side * crop_ratio))
    crops: List[Image.Image] = [img]

    left = (w - crop_size) // 2
    top = (h - crop_size) // 2

    points = [
        (left, top),                    # center
        (0, 0),                         # top-left
        (w - crop_size, 0),             # top-right
        (0, h - crop_size),             # bottom-left
        (w - crop_size, h - crop_size), # bottom-right
    ]

    for x, y in points:
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
def predict_probs_multicrop(img: Image.Image, crop_ratio: float = 0.80) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
    crops = get_crops(img, crop_ratio=crop_ratio)

    probs_list: List[torch.Tensor] = []
    per_crop_top: List[Tuple[str, float]] = []

    for c in crops:
        probs = predict_probs_single(c)
        probs_list.append(probs)

        probs_np = probs.detach().cpu().numpy()
        top_idx = int(np.argmax(probs_np))
        per_crop_top.append((CLASSES[top_idx], float(probs_np[top_idx])))

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (C,)
    return avg_probs, per_crop_top


# -----------------------------
# DECISION LOGIC
# -----------------------------
def compute_decision(prob_map: Dict[str, float]) -> Dict[str, Any]:
    malignancy_score = float(sum(prob_map.get(c, 0.0) for c in MALIGNANT))
    benign_score = float(1.0 - malignancy_score)

    # decision with uncertainty band
    if UNCERTAIN_LOW <= malignancy_score <= UNCERTAIN_HIGH:
        decision = "uncertain"
    else:
        decision = "malignant" if malignancy_score >= THRESHOLD else "benign"

    if malignancy_score >= HIGH_RISK:
        risk_level = "high"
    elif malignancy_score >= THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "decision": decision,
        "risk_level": risk_level,
        "malignancy_score": malignancy_score,
        "benign_score": benign_score,
        "threshold": THRESHOLD,
        "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
        "high_risk_cutoff": HIGH_RISK,
    }


# -----------------------------
# INFERENCE
# -----------------------------
@torch.no_grad()
def run_inference(img: Image.Image) -> Dict[str, Any]:
    # Quality gate first
    q = image_quality_check(img)
    if q is not None:
        return {"status": "bad_image", **q}

    probs, _ = predict_probs_multicrop(img, crop_ratio=CROP_RATIO)

    if probs.numel() != len(CLASSES):
        return {
            "status": "error",
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

    decision_pack = compute_decision(prob_map)

    return {
        "status": "ok",
        "top_class": top_class,
        "confidence": top_conf,
        "probabilities": prob_map,
        **decision_pack,
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
        "strategy": "multicrop_avg",
        "note": "Model confidence is not clinical probability.",
    }


@torch.no_grad()
def run_inference_debug(img: Image.Image) -> Dict[str, Any]:
    q = image_quality_check(img)
    if q is not None:
        return {"status": "bad_image", **q}

    probs, per_crop_top = predict_probs_multicrop(img, crop_ratio=CROP_RATIO)

    if probs.numel() != len(CLASSES):
        return {
            "status": "error",
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

    decision_pack = compute_decision(prob_map)

    # format per-crop info
    crop_list = []
    for i, (c, conf) in enumerate(per_crop_top):
        crop_list.append({"crop_index": i, "top_class": c, "confidence": conf})

    return {
        "status": "ok",
        "final": {
            "top_class": top_class,
            "confidence": top_conf,
            "probabilities": prob_map,
            **decision_pack,
        },
        "per_crop": crop_list,
        "meta": {
            "crops_used": len(per_crop_top),
            "crop_ratio": CROP_RATIO,
            "device": device,
            "model_dtype": str(MDTYPE).replace("torch.", ""),
        },
        "note": "Debug shows per-crop predictions and averaged result.",
    }


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_dtype": str(MDTYPE).replace("torch.", ""),
        "weights_path": WEIGHTS_PATH,
        "classes": CLASSES,
        "malignant_classes": sorted(list(MALIGNANT)),
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
        "high_risk_cutoff": HIGH_RISK,
        "strategy": "multicrop_avg",
        "quality_gates": {
            "min_side_px": MIN_SIDE_PX,
            "dark_mean_min": DARK_MEAN_MIN,
            "blur_var_min": BLUR_VAR_MIN,
        },
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        if not img_bytes:
            return {"status": "error", "error": "Empty file"}

        img = Image.open(io.BytesIO(img_bytes))
        return run_inference(img)

    except Exception as e:
        logger.exception("Analyze failed")
        return {"status": "error", "error": "Analyze failed", "detail": str(e)}


@app.post("/analyze_debug")
async def analyze_debug(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        img_bytes = await file.read()
        if not img_bytes:
            return {"status": "error", "error": "Empty file"}

        img = Image.open(io.BytesIO(img_bytes))
        return run_inference_debug(img)

    except Exception as e:
        logger.exception("Analyze_debug failed")
        return {"status": "error", "error": "Analyze_debug failed", "detail": str(e)}
