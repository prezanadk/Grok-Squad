import io
import os
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from typing import Dict, Any, List

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms, models

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-api")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ======================
# CONFIG
# ======================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your classes: akiec,bcc,bkl,df,mel,nv,vasc
MALIGNANT = {"mel", "bcc", "akiec"}

# Step 8 thresholds
THRESHOLD = 0.50
HIGH_RISK = 0.75

# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="Skin Lesion Classifier API (EfficientNet)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon mode
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# TRANSFORMS
# ======================
infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ======================
# LOAD MODEL
# ======================
def build_model(num_classes: int):
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        m = models.efficientnet_b0(weights=weights)
    except Exception:
        m = models.efficientnet_b0(pretrained=True)

    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model at: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES: List[str] = checkpoint["class_names"]

model = build_model(len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

logger.info(f"Loaded model on {DEVICE} with classes: {CLASS_NAMES}")

# ======================
# HELPERS
# ======================
@torch.no_grad()
def predict_pil(img: Image.Image) -> Dict[str, Any]:
    img = img.convert("RGB")
    x = infer_tfms(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]  # (C,)

    conf, idx = torch.max(probs, dim=0)
    top_class = CLASS_NAMES[idx.item()]
    top_conf = float(conf.item())

    prob_map = {
        cls: float(p)
        for cls, p in zip(CLASS_NAMES, probs.detach().cpu().tolist())
    }

    # malignancy score = sum probs of malignant classes
    malignancy_score = float(sum(prob_map.get(c, 0.0) for c in MALIGNANT))
    benign_score = float(1.0 - malignancy_score)

    decision = "malignant" if malignancy_score >= THRESHOLD else "benign"

    if malignancy_score >= HIGH_RISK:
        risk_level = "high"
    elif malignancy_score >= THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        # match frontend
        "top_class": top_class,
        "confidence": top_conf,
        "decision": decision,
        "risk_level": risk_level,
        "threshold": THRESHOLD,
        "high_risk_cutoff": HIGH_RISK,
        "cancerous": top_class in MALIGNANT,
        "malignancy_score": malignancy_score,
        "benign_score": benign_score,
        "probabilities": prob_map,
        "device": DEVICE,
        "model": "efficientnet_b0",
        "note": "Scores are model confidence, not clinical probability.",
    }

# ======================
# ROUTES
# ======================
@app.get("/")
def root():
    return {
        "status": "ok",
        "device": DEVICE,
        "classes": CLASS_NAMES,
        "malignant_classes": sorted(list(MALIGNANT)),
        "threshold": THRESHOLD,
        "high_risk_cutoff": HIGH_RISK,
    }
    
class ExplainRequest(BaseModel):
    lesion_type: str
    symptoms: str

@app.post("/ai_explain")
async def ai_explain(req: ExplainRequest):
    if not GEMINI_API_KEY:
        return {"error": "Missing GEMINI_API_KEY in backend/.env"}

    lesion = (req.lesion_type or "").strip().lower()
    symptoms = (req.symptoms or "").strip()

    # keep it safe + educational
    prompt = f"""
You are a medical education assistant for a hackathon app.
The model predicted lesion code: {lesion}

User-reported symptoms:
{symptoms}

Task:
- Explain what the lesion code usually refers to in simple language.
- List common symptoms that can be associated with it (general education).
- Provide red-flag symptoms that require urgent medical attention.
- Give safe photo-taking tips for better analysis (lighting, focus, distance).
- Add a clear disclaimer: "This is not a diagnosis."

Rules:
- Do NOT confirm cancer.
- Do NOT provide treatment instructions, prescriptions, or medication dosing.
- Encourage seeing a dermatologist if concerned.
- Keep the answer under 160 words.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return {"error": "Empty reply from Gemini"}
        return {"reply": text}
    except Exception as e:
        logger.exception("Gemini call failed")
        return {"error": "Gemini call failed", "detail": str(e)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            return {"error": "Empty file"}

        img = Image.open(io.BytesIO(content))
        return predict_pil(img)
    except Exception as e:
        logger.exception("Analyze failed")
        return {"error": "Analyze failed", "detail": str(e)}
