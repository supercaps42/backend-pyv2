import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="BananaVision API", description="AI-powered banana disease detection", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_penyakit_pisang.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Disease mapping
DISEASE_MAP = {
    0: {'name': 'Black Sigatoka', 'category': 'Jamur', 'severity': 'Berat'},
    1: {'name': 'Bract Mosaic Virus', 'category': 'Virus', 'severity': 'Sedang'},
    2: {'name': 'Healthy Leaf', 'category': 'Sehat', 'severity': 'Ringan'},
    3: {'name': 'Insect Pest', 'category': 'Hama', 'severity': 'Sedang'},
    4: {'name': 'Moko Disease', 'category': 'Bakteri', 'severity': 'Berat'},
    5: {'name': 'Panama Disease', 'category': 'Jamur', 'severity': 'Berat'},
    6: {'name': 'Yellow Sigatoka', 'category': 'Jamur', 'severity': 'Sedang'},
}

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    image: str  # base64 encoded image

class PredictionResult(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: Optional[str] = None

def preprocess_image(image_data, target_size=(224, 224)):
    """Convert base64 or PIL image to preprocessed array"""
    if isinstance(image_data, str):
        # Base64 string
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = image_data

    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """ML prediction endpoint"""
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="No image provided")

        # Preprocess image
        image_array = preprocess_image(request.image)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        confidence_scores = predictions[0]
        predicted_class = np.argmax(confidence_scores)
        confidence = float(confidence_scores[predicted_class]) * 100

        # Map to disease info
        disease_info = DISEASE_MAP.get(predicted_class, {
            'name': 'Unknown',
            'category': 'Unknown',
            'severity': 'Unknown'
        })

        # Return results
        return PredictionResponse(
            success=True,
            data={
                'detectedDisease': disease_info['name'],
                'category': disease_info['category'],
                'severity': disease_info['severity'],
                'confidence': round(confidence, 2),
                'predictions': [
                    {
                        'disease': DISEASE_MAP.get(i, {}).get('name', f'Class {i}'),
                        'confidence': round(float(confidence_scores[i]) * 100, 2)
                    }
                    for i in range(len(confidence_scores))
                ]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')

@app.post("/api/predict-file", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """ML prediction endpoint with file upload"""
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Preprocess image
        image_array = preprocess_image(img)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        confidence_scores = predictions[0]
        predicted_class = np.argmax(confidence_scores)
        confidence = float(confidence_scores[predicted_class]) * 100

        # Map to disease info
        disease_info = DISEASE_MAP.get(predicted_class, {
            'name': 'Unknown',
            'category': 'Unknown',
            'severity': 'Unknown'
        })

        # Return results
        return PredictionResponse(
            success=True,
            data={
                'detectedDisease': disease_info['name'],
                'category': disease_info['category'],
                'severity': disease_info['severity'],
                'confidence': round(confidence, 2),
                'predictions': [
                    {
                        'disease': DISEASE_MAP.get(i, {}).get('name', f'Class {i}'),
                        'confidence': round(float(confidence_scores[i]) * 100, 2)
                    }
                    for i in range(len(confidence_scores))
                ]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}

@app.get("/")
async def root():
    return {"message": "BananaVision API", "version": "1.0.0", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
