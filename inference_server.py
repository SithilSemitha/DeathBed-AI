"""
Inference server for the trained decision classifier
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import uvicorn
import os

print("=" * 60)
print("DEATHBED - INFERENCE SERVER")
print("=" * 60)

# Load model
model_path = 'models/decision_classifier.pkl'
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("   Run 'python train_from_csv.py' first!")
    exit(1)

model = joblib.load(model_path)
print(f"✅ Model loaded from {model_path}")
print(f"   Categories: {list(model.classes_)}")

# Load labels
labels_path = 'models/labels.pkl'
labels = joblib.load(labels_path) if os.path.exists(labels_path) else list(model.classes_)

app = FastAPI(title="DeathBed Decision Classifier", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DecisionText(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    category: str
    subCategory: str
    confidence: float
    summary: str
    requiresOverride: bool

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_subcategory(category, text):
    text_lower = text.lower()
    mapping = {
        'career': {
            'quit': 'Quit Job',
            'start': 'Start Company',
            'leave': 'Leave Job',
            'change': 'Career Change',
            'default': 'Career Move'
        },
        'education': {
            'major': 'Choose Major',
            'degree': 'Choose Degree',
            'drop': 'Dropout Decision',
            'default': 'Education Decision'
        },
        'relationship': {
            'propos': 'Marriage Decision',
            'marry': 'Marriage Decision',
            'break': 'Breakup Decision',
            'default': 'Relationship Decision'
        },
        'finance': {
            'buy': 'Purchase Decision',
            'invest': 'Investment',
            'loan': 'Loan Decision',
            'default': 'Financial Decision'
        },
        'relocation': {
            'move': 'Relocation',
            'relocate': 'Relocation',
            'default': 'Relocation'
        },
        'family': {
            'children': 'Have Children',
            'parent': 'Parenting Decision',
            'default': 'Family Decision'
        },
        'health': {
            'exercise': 'Health Decision',
            'diet': 'Health Decision',
            'default': 'Health Decision'
        },
        'lifestyle': {
            'minimal': 'Lifestyle Change',
            'habit': 'Lifestyle Change',
            'default': 'Lifestyle Change'
        }
    }
    for keyword, subcat in mapping.get(category, {}).items():
        if keyword in text_lower and keyword != 'default':
            return subcat
    return mapping.get(category, {}).get('default', 'General Decision')

@app.post("/classify", response_model=ClassificationResponse)
async def classify(item: DecisionText):
    if len(item.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Minimum 20 characters required")
    
    cleaned = clean_text(item.text)
    probs = model.predict_proba([cleaned])[0]
    idx = probs.argmax()
    category = model.classes_[idx]
    confidence = float(probs[idx])
    requires_override = confidence < 0.60

    sub_category = generate_subcategory(category, item.text)
    summary = item.text[:120] + ("..." if len(item.text) > 120 else "")

    return {
        "category": category,
        "subCategory": sub_category,
        "confidence": confidence,
        "summary": summary,
        "requiresOverride": requires_override
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": "logistic_regression"}

if __name__ == "__main__":
    port = int(os.getenv("ML_PORT", 8000))
    print(f"\n🚀 Starting server on port {port}...")
    print(f"   Health: http://localhost:{port}/health")
    print(f"   Classify: POST http://localhost:{port}/classify")
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")