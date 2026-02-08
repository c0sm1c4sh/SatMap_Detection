from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
import os
import sys

# Ensure we can import model_utils
sys.path.append(os.path.dirname(__file__))
from model_utils import get_model, get_transform, get_device

app = FastAPI()

# 1. Setup Device & Model
device = get_device()
CLASS_LABELS = ['Forest', 'Herbaceous Vegetation', 'Permanent Crop', 'Residential', 'Industrial']

# Use the centralized model helper
model = get_model(num_classes=5)

# Load weights correctly for the available device
weights_path = 'weights/model.pth'
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print(f"⚠️ Warning: {weights_path} not found. Running with random weights.")

model.to(device)
model.eval()

# 2. Get standard transform
preprocess = get_transform()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    
    # Preprocess and MOVE to same device as model
    img_t = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(img_t)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)
        
    return {
        "prediction": CLASS_LABELS[idx.item()], 
        "confidence": round(conf.item(), 4),
        "device_used": str(device)
    }