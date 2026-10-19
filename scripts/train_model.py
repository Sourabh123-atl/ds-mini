#!/usr/bin/env python3
"""TF-IDF + Logistic Regression 3-class sentiment model on the hand-labelled
Coronavirus Tweets NLP dataset (Kaggle datatattle, HF mirror argilla/twitter-coronavirus).
Uses the dataset's own train/test split. Exports models/sentiment_tfidf_lr.joblib + metrics.json.
"""
import json
from pathlib import Path
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MAP = {"Extremely Negative": "Negative", "Negative": "Negative", "Neutral": "Neutral",
       "Positive": "Positive", "Extremely Positive": "Positive"}

d = pd.read_parquet(ROOT / "data/raw/corona_nlp.parquet")
d["label5"] = d.prediction.map(lambda p: p[0]["label"])
d["y"] = d.label5.map(MAP)
d["split"] = d.metadata.map(lambda m: m["split"])
d["x"] = d.text.str.replace(r"https?://\S+|@\w+", " ", regex=True)
tr, te = d[d.split == "train"], d[d.split == "test"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
])
pipe.fit(tr.x, tr.y)
pred = pipe.predict(te.x)
labels = ["Negative", "Neutral", "Positive"]
m = {
    "dataset": "Coronavirus Tweets NLP (datatattle, Kaggle) via https://huggingface.co/datasets/argilla/twitter-coronavirus",
    "task": "3-class sentiment (Extremely Negative+Negative / Neutral / Positive+Extremely Positive)",
    "train_size": len(tr), "test_size": len(te),
    "accuracy": round(accuracy_score(te.y, pred), 4),
    "f1_macro": round(f1_score(te.y, pred, average="macro"), 4),
    "f1_per_class": {k: round(v, 4) for k, v in zip(labels, f1_score(te.y, pred, average=None, labels=labels))},
    "confusion_matrix_labels": labels,
    "confusion_matrix": confusion_matrix(te.y, pred, labels=labels).tolist(),
    "report": classification_report(te.y, pred, labels=labels, output_dict=True),
}
# Baseline: VADER on the same test set
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
vp = te.x.map(lambda s: (lambda c: "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral")(sia.polarity_scores(s)["compound"]))
m["vader_baseline_accuracy"] = round(accuracy_score(te.y, vp), 4)
m["vader_baseline_f1_macro"] = round(f1_score(te.y, vp, average="macro"), 4)
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(pipe, ROOT / "models/sentiment_tfidf_lr.joblib")
(ROOT / "models/metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps({k: v for k, v in m.items() if k != "report"}, indent=1))

# Apply model to the Preda tweets for the dashboard
tw = pd.read_parquet(ROOT / "data/processed/tweets.parquet")
tw["model_sentiment"] = pipe.predict(tw.text.str.replace(r"https?://\S+|@\w+", " ", regex=True))
tw.to_parquet(ROOT / "data/processed/tweets.parquet", index=False)
print("agreement model vs VADER on COVID19 Tweets:", round((tw.model_sentiment == tw.sentiment).mean(), 4))
print(tw.model_sentiment.value_counts(normalize=True).round(4).to_dict())
