from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

# If using TensorFlow:
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary: load model if it exists, otherwise None
MODEL_PATH = "model.h5"
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception:
    model = None

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def bucket(p: float):
    if p < 0.4: return "Low"
    if p < 0.7: return "Medium"
    return "High"

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return {"error": "Upload a JPG or PNG image."}

    content = await file.read()
    img = Image.open(io.BytesIO(content))
    x = preprocess(img)

    if model is None:
        # Demo fallback if model isn't ready yet
        return {
            "risk_level": "Unknown",
            "confidence": None,
            "message": "Model not loaded yet. Train and place model.h5 in backend/.",
            "disclaimer": "Not a medical diagnosis."
        }

    prob = float(model.predict(x)[0][0])
    return {
        "risk_level": bucket(prob),
        "confidence": round(prob * 100, 2),
        "recommendation": "Consult a dermatologist if risk is Medium/High.",
        "disclaimer": "Not a medical diagnosis. Screening support only."
    }
