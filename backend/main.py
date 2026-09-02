import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import json
from ml.preprocessing import clean_text

app = FastAPI(title='Customer Complaints Classifier')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get('FRONTEND_ORIGINS', '*')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    complaint:str


MODEL_PATH = os.environ.get('MODEL_PATH', '../models')


def load_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    # allow override
    model_path = os.path.abspath(os.environ.get('MODEL_PATH', model_path))
    try:
        model = joblib.load(os.path.join(model_path, 'final_model.joblib'))
        vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.joblib'))
        label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
        # optional results.json
        results = None
        results_path = os.path.join(model_path, 'results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
            except Exception:
                results = None
    except Exception as e:
        raise RuntimeError(f'Failed loading model artifacts: {e}')
    return model, vectorizer, label_encoder, results


try:
    model, vectorizer, label_encoder, results_json = load_artifacts()
except Exception as e:
    model = None
    vectorizer = None
    label_encoder = None
    load_error = str(e)


@app.get('/')
def health():
    if model is None:
        return {'status': 'error', 'detail': load_error}
    return {'status': 'ok', 'model_loaded': True}


@app.post('/predict')
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')
    text = payload.complaint
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail='Empty complaint provided')
    try:
        # apply same cleaning as training
        text_clean = clean_text(text)
        x = vectorizer.transform([text_clean])
        # predict
        prediction = None
        confidence = None
        # If model supports probabilities (e.g., LogisticRegression, Naive Bayes)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(x)
            # probs shape: (n_samples, n_classes)
            probs_arr = np.asarray(probs)
            # take first sample
            sample_probs = probs_arr[0]
            idx = int(np.argmax(sample_probs))
            # inverse transform expects array-like of encoded labels
            prediction = label_encoder.inverse_transform([idx])[0]
            confidence = float(sample_probs[idx])
        elif hasattr(model, 'decision_function'):
            df = model.decision_function(x)
            df_arr = np.asarray(df)
            # df_arr shape can be (n_classes,) or (1, n_classes) or (n_samples, n_classes)
            # normalize to 1D array of class scores for first sample
            if df_arr.ndim == 1:
                scores = df_arr
            else:
                scores = df_arr[0]
            # binary classifier produces single score; map through sigmoid
            if scores.size == 1:
                score = float(scores.item())
                prob = 1.0 / (1.0 + float(np.exp(-score)))
                idx = int(prob >= 0.5)
                prediction = label_encoder.inverse_transform([idx])[0]
                confidence = float(prob)
            else:
                # multiclass scores -> softmax
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
                idx = int(np.argmax(probs))
                prediction = label_encoder.inverse_transform([idx])[0]
                confidence = float(probs[idx])
        else:
            pred = model.predict(x)
            pred_arr = np.asarray(pred)
            pred0 = pred_arr[0]
            # pred0 may already be encoded label
            prediction = label_encoder.inverse_transform([int(pred0)])[0]
            confidence = None

        # Build response, include confidence only if available
        resp = {'prediction': prediction}
        if confidence is not None:
            resp['confidence'] = confidence
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {e}')


@app.get('/metrics')
def metrics():
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')
    if results_json is None:
        raise HTTPException(status_code=404, detail='Results not available')
    return results_json
