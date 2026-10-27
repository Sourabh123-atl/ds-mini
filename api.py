"""COVID-19 Social Surveillance & Sentiment REST API.
Built with FastAPI, Scikit-learn, and VADER Sentiment.

Run standalone:
    python api.py
OR
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Interactive API Docs:
    http://localhost:8000/docs
"""
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

app = FastAPI(
    title="COVID-19 Social Surveillance & NLP Sentiment API",
    description="Production REST API providing real-time sentiment inference, XAI token attribution, and epidemic anomaly detection.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- In-Memory Resources -----------------
model_pipeline = None
vader_analyzer = None
summary_data = {}
daily_data = None
lag_data = None


@app.on_event("startup")
def load_resources():
    global model_pipeline, vader_analyzer, summary_data, daily_data, lag_data
    
    # 1. Load ML Model
    model_path = MODELS_DIR / "sentiment_tfidf_lr.joblib"
    if model_path.exists():
        try:
            model_pipeline = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Failed to load ML model: {e}")
            
    # 2. VADER
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # 3. Datasets
    summ_path = PROCESSED_DIR / "summary.json"
    if summ_path.exists():
        summary_data = json.loads(summ_path.read_text())
        
    daily_path = PROCESSED_DIR / "daily.csv"
    if daily_path.exists():
        daily_data = pd.read_csv(daily_path, parse_dates=["day"])
        
    lag_path = PROCESSED_DIR / "lag_corr.csv"
    if lag_path.exists():
        lag_data = pd.read_csv(lag_path)


# Eagerly load resources so endpoints are available in all ASGI runners
load_resources()


# ----------------- Pydantic Schemas -----------------
class PredictRequest(BaseModel):
    text: str = Field(..., example="Hospitals in Mumbai are facing severe shortage of oxygen cylinders")
    explain: bool = Field(default=True, description="Compute XAI token attributions")


class TokenAttribution(BaseModel):
    token: str
    weight: float
    impact: str  # 'supportive' or 'opposing'


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    vader_sentiment: str
    vader_compound: float
    vader_scores: Dict[str, float]
    consensus: bool
    top_explanations: Optional[List[TokenAttribution]] = None


class BatchPredictRequest(BaseModel):
    tweets: List[str] = Field(..., max_items=500)


class AnomalyItem(BaseModel):
    date: str
    metric: str
    value: float
    baseline_mean: float
    z_score: float
    severity: str


# ----------------- Endpoints -----------------
@app.get("/", tags=["Health & Status"])
def root():
    return {
        "status": "online",
        "service": "COVID-19 Surveillance & Sentiment API",
        "version": "2.0.0",
        "model_loaded": model_pipeline is not None,
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health & Status"])
def health_check():
    return {
        "status": "healthy",
        "model": "active" if model_pipeline is not None else "degraded",
        "vader": "active"
    }


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
def predict_sentiment(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    cleaned_text = re.sub(r"https?://\S+|@\w+", " ", req.text)
    
    # 1. ML Pipeline Prediction
    sentiment = "Neutral"
    confidence = 0.5
    probs_dict = {"Negative": 0.333, "Neutral": 0.334, "Positive": 0.333}
    top_attributions = None
    
    if model_pipeline is not None:
        try:
            probs = model_pipeline.predict_proba([cleaned_text])[0]
            classes = model_pipeline.classes_
            probs_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
            best_idx = probs.argmax()
            sentiment = str(classes[best_idx])
            confidence = round(float(probs[best_idx]), 4)
            
            # Compute Explainable AI (XAI) Token Attribution
            if req.explain:
                tfidf = model_pipeline.named_steps["tfidf"]
                clf = model_pipeline.named_steps["clf"]
                vec = tfidf.transform([cleaned_text])
                feats = tfidf.get_feature_names_out()
                row = vec.tocoo()
                
                contribs = []
                for col in row.col:
                    w = float(vec[0, col] * clf.coef_[best_idx, col])
                    contribs.append(TokenAttribution(
                        token=str(feats[col]),
                        weight=round(w, 4),
                        impact="supportive" if w > 0 else "opposing"
                    ))
                contribs.sort(key=lambda x: abs(x.weight), reverse=True)
                top_attributions = contribs[:8]
        except Exception as e:
            print(f"Inference error: {e}")
            
    # 2. VADER Baseline
    vader_res = vader_analyzer.polarity_scores(req.text)
    comp = round(vader_res["compound"], 4)
    v_sentiment = "Positive" if comp >= 0.05 else ("Negative" if comp <= -0.05 else "Neutral")
    
    return PredictResponse(
        text=req.text,
        sentiment=sentiment,
        confidence=confidence,
        probabilities=probs_dict,
        vader_sentiment=v_sentiment,
        vader_compound=comp,
        vader_scores=vader_res,
        consensus=(sentiment == v_sentiment),
        top_explanations=top_attributions
    )


@app.post("/api/v1/batch-predict", tags=["Inference"])
def batch_predict(req: BatchPredictRequest):
    if not req.tweets:
        return {"predictions": []}
        
    cleaned = [re.sub(r"https?://\S+|@\w+", " ", t) for t in req.tweets]
    results = []
    
    if model_pipeline is not None:
        preds = model_pipeline.predict(cleaned)
        probs = model_pipeline.predict_proba(cleaned)
        for t, pred, pr in zip(req.tweets, preds, probs):
            results.append({
                "tweet": t,
                "sentiment": str(pred),
                "confidence": round(float(pr.max()), 4)
            })
    else:
        for t in req.tweets:
            comp = vader_analyzer.polarity_scores(t)["compound"]
            lab = "Positive" if comp >= 0.05 else ("Negative" if comp <= -0.05 else "Neutral")
            results.append({
                "tweet": t,
                "sentiment": lab,
                "confidence": round(abs(comp), 4)
            })
            
    return {"total": len(results), "predictions": results}


@app.get("/api/v1/surveillance/anomalies", response_model=List[AnomalyItem], tags=["Epidemiological Surveillance"])
def detect_anomalies(
    threshold: float = Query(default=1.5, ge=1.0, le=4.0, description="Z-Score standard deviation threshold"),
    metric: str = Query(default="india", description="Metric to test: 'india', 'maha', 'mumbai', 'mean_compound'")
):
    if daily_data is None or metric not in daily_data.columns:
        raise HTTPException(status_code=404, detail="Surveillance daily series unavailable.")
        
    s = daily_data[metric].astype(float)
    rm = s.rolling(5, min_periods=2).mean()
    rstd = s.rolling(5, min_periods=2).std().replace(0, np.nan).fillna(1.0)
    z_scores = (s - rm) / rstd
    
    anomalies = []
    for idx, z in z_scores.items():
        if pd.notna(z) and z >= threshold:
            val = float(s.iloc[idx])
            mean_val = float(rm.iloc[idx])
            sev = "Critical Outbreak Alert" if z >= 2.5 else ("Warning" if z >= 2.0 else "Elevated Watch")
            anomalies.append(AnomalyItem(
                date=str(daily_data["day"].iloc[idx].date()),
                metric=metric,
                value=round(val, 2),
                baseline_mean=round(mean_val, 2),
                z_score=round(float(z), 2),
                severity=sev
            ))
            
    return anomalies


@app.get("/api/v1/surveillance/summary", tags=["Epidemiological Surveillance"])
def get_surveillance_summary():
    return summary_data


@app.get("/api/v1/surveillance/correlations", tags=["Epidemiological Surveillance"])
def get_lag_correlations():
    if lag_data is None:
        raise HTTPException(status_code=404, detail="Lag correlation dataset unavailable.")
    return lag_data.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting COVID-19 Surveillance REST API on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
