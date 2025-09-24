import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from src.data_processing import INV_LABEL_MAP

def load_baseline(path='artifacts/baseline_model.pkl'):
    return joblib.load(path)

def load_hf_model(model_dir='artifacts/hf_model', model_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir if model_name is None else model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir if model_name is None else model_name)
    return tokenizer, model

def hf_predict(texts, tokenizer, model):
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)
    labels = [INV_LABEL_MAP[int(p)] for p in preds]
    confidences = [float(probs[i, preds[i]]) for i in range(len(preds))]
    return labels, confidences
