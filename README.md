DERMAL LESION INTELLIGENCE
Project by Team GROK SQUAD

PROJECT OVERVIEW

Dermal Lesion Intelligence is a skin lesion classification system developed by Team Grok Squad to assist in early skin cancer risk assessment using medical images.

The system analyzes an uploaded skin lesion image and predicts:

The lesion type

Whether it is cancerous or non-cancerous

The confidence level of the prediction

This project was built for a hackathon and is intended for educational and research purposes only.
It is NOT a medical diagnostic tool.

TEAM

Team Name: Grok Squad
Project Name: Dermal Lesion Intelligence

MODEL DETAILS

Framework: PyTorch
Backend Model Format: .pth

The backend uses a PyTorch model loaded from a .pth file (state dictionary).
YOLO .pt files were explored earlier during experimentation, but the FINAL backend inference uses a .pth model for better control and flexibility.

SUPPORTED LESION CLASSES

akiec - Actinic Keratoses (Cancerous)
bcc - Basal Cell Carcinoma (Cancerous)
mel - Melanoma (Cancerous)

bkl - Benign Keratosis (Non-Cancerous)
nv - Melanocytic Nevus (Non-Cancerous)
df - Dermatofibroma (Non-Cancerous)
vasc - Vascular Lesion (Non-Cancerous)

DATASET

The model was trained using publicly available dermatology datasets, including:

Roboflow/north south university

ISIC Archive

Images were preprocessed and augmented.
Class imbalance was handled during training.

PROJECT STRUCTURE

Dermal-Lesion-Intelligence/
backend/
main.py Backend API
model.py PyTorch model architecture
best_model.pth Trained model weights
data/
train/
val/
frontend/
train.py Training script
eval.py Evaluation script
README.txt

RUNNING THE BACKEND

Install dependencies:
pip install -r requirements.txt

Start the server:
uvicorn main:app --host 0.0.0.0 --port 8000

Open in browser:
http://localhost:8000/docs

Upload a skin lesion image to the /analyze endpoint.

EXAMPLE OUTPUT

Predicted Class: mel
Description: Melanoma (Cancerous)
Confidence: 0.82
Risk Level: High

LIMITATIONS

Not clinically validated

Performance depends on image quality and lighting

Dataset may contain skin-tone bias

External or non-medical images may reduce accuracy

FUTURE IMPROVEMENTS

More diverse skin tone data

Out-of-distribution image detection

Explainable visualizations

Mobile application support

Doctor-in-the-loop validation

DISCLAIMER

This software does NOT replace professional medical advice.
Always consult a certified dermatologist for diagnosis and treatment.
