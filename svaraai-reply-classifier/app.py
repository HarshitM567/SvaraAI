from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import traceback

from src.model_utils import load_baseline, load_hf_model, hf_predict

app = FastAPI(title='SvaraAI Reply Classifier')

class RequestIn(BaseModel):
    text: str

BASELINE_PATH = 'artifacts/baseline_model.pkl'
HF_MODEL_DIR = 'artifacts/hf_model'

baseline = None
hf_tokenizer = None
hf_model = None

@app.on_event('startup')
def load_models():
    global baseline, hf_tokenizer, hf_model
    try:
        baseline = load_baseline(BASELINE_PATH)
    except Exception:
        baseline = None
    try:
        hf_tokenizer, hf_model = load_hf_model(HF_MODEL_DIR)
    except Exception:
        hf_tokenizer = None
        hf_model = None

@app.post('/predict')
async def predict(payload: RequestIn):
    text = payload.text
    try:
        if hf_tokenizer is not None and hf_model is not None:
            labels, confidences = hf_predict([text], hf_tokenizer, hf_model)
            return {'label': labels[0], 'confidence': confidences[0]}
        elif baseline is not None:
            pred = baseline.predict([text])[0]
            proba = max(baseline.predict_proba([text])[0]) if hasattr(baseline, 'predict_proba') else None
            label_map = {0:'positive',1:'negative',2:'neutral'}
            label = label_map.get(int(pred), str(pred))
            return {'label': label, 'confidence': float(proba) if proba is not None else None}
        else:
            return {'error': 'no model available. please run training and place artifacts in artifacts/'}
    except Exception as e:
        return {'error': 'prediction failed', 'details': traceback.format_exc()}
