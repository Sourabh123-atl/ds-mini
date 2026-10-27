# 🦠 COVID / Influenza Trend Analysis Using Social Media Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

A comprehensive Data Science & Natural Language Processing project analyzing **178,000+ COVID-19 tweets** with an **India / Maharashtra / Mumbai** focus. The pipeline correlates social media discussion volume, clinical symptom keywords, and public sentiment against official COVID-19 epidemiological case curves (covid19india.org & Our World in Data).

Includes an interactive **Streamlit dashboard**, a supervised **TF-IDF + Logistic Regression sentiment classifier** (78.2% test accuracy), **NMF Topic Modeling**, and **Lag-Cross-Correlation analysis**.

---

## 🚀 Live Deployment Guide

The project is pre-configured for zero-friction 1-click cloud deployment:

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Fork or push this repository to your GitHub account (`https://github.com/<your-username>/ds-mini`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository and `main` branch.
4. Set **Main file path** to: `streamlit_app.py` (or `app/streamlit_app.py`).
5. Click **"Deploy!"** — it will build and launch in under 2 minutes.

### Option 2: Render.com (Web Service)
This repo contains a pre-configured `render.yaml` and `Procfile`:
1. Log into [Render.com](https://dashboard.render.com/).
2. Click **"New +"** $\rightarrow$ **"Web Service"** and connect your GitHub repo.
3. Configure the settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment Variable**: `PYTHON_VERSION = 3.11.9`
4. Click **"Create Web Service"**.

### Option 3: Docker Container
Run the fully-contained Docker image anywhere (Docker Desktop, Hugging Face Spaces, AWS, GCP, Azure):
```bash
# Build the Docker image
docker build -t covid-trends-app .

# Run container on port 8501
docker run -p 8501:8501 covid-trends-app
```
Access the application at `http://localhost:8501`.

---

## 🌟 Advanced Platform Capabilities

| Tab / Module | Feature | Advanced Innovation |
|---|---|---|
| 📈 **Surveillance & Cases** | Epidemic Curve Surveillance | Dual-axis tracking of daily tweet chatter vs confirmed cases in India, Maharashtra, and Mumbai. |
| 😊 **Sentiment Dynamics** | Longitudinal Polarity Tracking | Daily sentiment composition (Negative / Neutral / Positive), 3-day rolling compound score, and top regional exemplars. |
| 🚨 **Anomaly & Outbreak Alerts** | Statistical Anomaly Engine | Rolling $Z$-Score surveillance ($1.5\sigma - 3.0\sigma$) detecting abnormal surges in clinical chatter and public panic. |
| 🔮 **Scenario Simulator** | What-If Projection Simulator | Interactive epidemiological scenario simulator predicting case trajectory shifts based on empirical lag elasticity. |
| 🧩 **NMF Topic Discovery** | Unsupervised Semantic Mining | 8 semantic topic themes extracted via Non-Negative Matrix Factorization with longitudinal share evolution. |
| 🔑 **Keywords & Co-occurrence** | Clinical Surveillance Network | Symptom prevalence rates & pairwise clinical term co-occurrence heatmap matrix across 178k tweets. |
| 📊 **Lead/Lag Correlation** | Cross-Correlation Surveillance | Pearson correlation across lead/lag offsets ($\tau \in [-7, +7]$ days) evaluating social signals as early warning indicators. |
| 🤖 **Live Model & XAI** | Explainable AI (XAI) Attribution | Live model comparator with token-level feature importance attribution visualizing exact words driving classification. |
| 📋 **Executive Briefing** | Situational Intelligence Report | One-click automated epidemiological intelligence briefing generation with instant Markdown export. |
| 🗂️ **Data Explorer** | High-Speed Corpus Explorer | Filter by keyword, region, date range, or sentiment class with high-speed memory-safeguarded CSV downloads. |
| 🔌 **Production REST API** | FastAPI Backend (`api.py`) | High-throughput REST API with live Swagger documentation (`/docs`) for real-time predictions and batch inference. |

---

## 🌐 FLUX Web Command Center & REST API (`api.py`)

In addition to the Streamlit UI, this platform features **FLUX — Infectious Disease Intelligence Command Center**, a modern single-page web application (SPA) backed by **FastAPI** (`http://localhost:8000`):

```bash
# Launch the FLUX Web Application & REST API
python api.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Command Center UI**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger REST Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Analytics Dashboard**: [http://localhost:8501/](http://localhost:8501/)

### Web Command Center Features:
1. **Academic / Healthcare Theme**: Grounded, clinical interface with dynamic Light / Dark Mode toggling and persistent user preferences.
2. **Interactive Data Ingestion**: Client-side CSV uploader allowing researchers to ingest and analyze custom social or clinical records on the fly.
3. **Clinical Symptom Co-occurrence Matrix**: Real-time pairwise symptom correlation tracking (Fever, Cough, Dyspnea, Fatigue, Loss of Smell).
4. **Interactive Leaflet GIS Map**: Geospatial outbreak and chatter distribution rendered across Indian states and global epicenters with interactive heat circles.
5. **Explainable AI (XAI) Attribution**: Word-level feature importance attribution visualizing exact vocabulary tokens driving sentiment and risk classifications.
6. **Outbreak Anomaly Engine**: Longitudinal $Z$-score anomaly flags detecting public panic and clinical discussion surges ($1.5\sigma - 3.0\sigma$).
7. **Research Workspace**: Client-persisted researcher notes, bookmarked case insights, and one-click Executive Briefing generator with Markdown export.

### REST API Endpoints:
- `GET /api/v1/overview` — High-level epidemiological corpus metrics, disease breakdown, and sentiment distributions.
- `GET /api/v1/surveillance/daily` — Daily time series combining tweet volumes, sentiment polarity, and official confirmed cases.
- `GET /api/v1/surveillance/anomalies` — Automated rolling $Z$-score outbreak anomaly detection flags.
- `GET /api/v1/surveillance/forecast` — 7-day trend projections with confidence intervals.
- `GET /api/v1/surveillance/geo` — GeoJSON-ready regional outbreak distribution data.
- `GET /api/v1/surveillance/cooccurrence` — Pairwise clinical symptom co-occurrence matrix.
- `GET /api/v1/surveillance/topics` — Precomputed 8-topic NMF semantic cluster distributions.
- `POST /api/v1/predict` — Single tweet inference with Explainable AI (XAI) token weights.
- `POST /api/v1/batch-predict` — High-speed vectorized batch prediction (up to 500 records).
- `GET /api/v1/models/evaluation` — Full model evaluation metrics, confusion matrix, and classification report.
- `POST /api/v1/reports/generate` — Automated epidemiological intelligence briefing generator.

---

## 🔬 How FLUX Works End-to-End (System Architecture)

```
[Raw Sources: Kaggle, OWID, covid19india.org]
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Data Cleaning & Spatial Normalization (analyze.py)   │
│    • Regex cleansing (URLs, handles, emojis, punct)     │
│    • Indian State & District Geotagging via Lexicon     │
│    • Clinical Symptom Flagging (12 WHO keywords)        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. NLP & Sentiment Pipeline (train_model.py)            │
│    • TF-IDF Vectorizer (1-2 n-grams, 50,000 features)   │
│    • Regularized Logistic Regression (C=4.0, balanced)  │
│    • VADER Rule-based Lexicon Sentiment (Compound)      │
│    • NMF Topic Modeling (8 Unsupervised Themes)         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Epidemiological Analytics & Anomaly Engine           │
│    • Pearson Lead/Lag Cross-Correlation (τ ∈ [-7, +7])  │
│    • Rolling Z-Score Anomaly Detector (1.5σ - 3.0σ)     │
│    • Dual-Axis Time Series Alignment with Case Curves   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. FastAPI High-Throughput REST Service (api.py)        │
│    • Static Single-Page App Hosting at /                │
│    • OpenAPI / Swagger Documentation at /docs           │
│    • Explainable AI Token Attribution Engine            │
└───────────────────┬───────────────────┬─────────────────┘
                    │                   │
                    ▼                   ▼
┌───────────────────────────────┐ ┌──────────────────────┐
│ FLUX Command Center Web App   │ │ Streamlit Analytics  │
│ (Vanilla JS + Chart.js + GIS) │ │ Dashboard (port 8501)│
│ Port 8000                     │ │                      │
└───────────────────────────────┘ └──────────────────────┘
```


## 📊 Model Evaluation & Benchmarks

The sentiment model was trained on the official benchmark train/test split of the **Coronavirus Tweets NLP Dataset** (44,955 tweets):

| Metric | TF-IDF + Logistic Regression | VADER Baseline |
|---|---|---|
| **Accuracy** | **78.20%** | 93.50% (Lexicon agreement) |
| **Macro F1-Score** | **0.7586** | 0.9296 |
| **Negative F1** | **0.8006** | — |
| **Neutral F1** | **0.6601** | — |
| **Positive F1** | **0.8150** | — |

*Model artifact stored at `models/sentiment_tfidf_lr.joblib` (~3.2 MB).*

---

## 📂 Repository Structure

```
ds-mini/
├── .streamlit/
│   └── config.toml             # Streamlit server, headless & theme configurations
├── app/
│   └── streamlit_app.py        # Main Streamlit analytics dashboard
├── data/
│   ├── processed/              # Curated, preprocessed datasets (ready to run)
│   │   ├── tweets.parquet      # 178k cleaned tweets with NLP & spatial tags
│   │   ├── cases_daily.csv     # Daily confirmed cases (India, MH, Mumbai, OWID)
│   │   ├── daily.csv           # Merged daily metrics & sentiment aggregates
│   │   ├── lag_corr.csv        # Pearson lead/lag correlation results
│   │   └── summary.json        # Precomputed corpus statistics & topic weights
│   └── raw/                    # Raw input sources (downloadable via script)
├── models/
│   ├── sentiment_tfidf_lr.joblib # Trained TF-IDF + Logistic Regression model
│   └── metrics.json            # Model performance, classification report & confusion matrix
├── scripts/
│   ├── analyze.py              # Data cleaning, spatial tagging, NMF & lag pipeline
│   ├── download_data.py        # Automated raw dataset fetcher
│   ├── screenshots.py          # Automated UI screenshot capture
│   └── train_model.py          # Sentiment model training & evaluation script
├── report/
│   └── figures/                # Visual charts for project documentation
├── Dockerfile                  # Container build instructions with health checks
├── Procfile                    # Process entrypoint for Heroku & Render
├── render.yaml                 # Infrastructure-as-code for Render deployment
├── requirements.txt            # Production dependencies for cloud hosting
├── requirements-dev.txt        # Optional developer dependencies (Playwright, Docx)
├── streamlit_app.py            # Root application entrypoint for Streamlit Cloud
└── README.md
```

---

## 💻 Local Installation & Setup

All preprocessed data and model weights are committed, so the app runs immediately after cloning without needing to download raw data:

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Sourabh123-atl/ds-mini.git
cd ds-mini

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔄 Rebuilding From Scratch (Optional)

If you wish to re-download the raw data (~119 MB) and re-train the models:
```bash
# 1. Download raw public datasets
python scripts/download_data.py

# 2. Run analysis, spatial extraction, and NMF decomposition (~20s)
python scripts/analyze.py

# 3. Train sentiment pipeline & compute metrics (~25s)
python scripts/train_model.py

# 4. Launch updated app
streamlit run streamlit_app.py
```

---

## 📑 Data Sources
- **COVID19 Tweets**: Gabriel Preda (`gpreda/covid19-tweets`), Kaggle (179,108 rows).
- **Coronavirus Tweets NLP**: DataTattle (`datatattle/covid-19-nlp-text-classification`), Hugging Face (44,955 rows).
- **India & District Cases**: [covid19india.org archive](https://data.covid19india.org/).
- **Global Cases**: [Our World in Data (OWID)](https://github.com/owid/covid-19-data).

---

## 👥 Authors & Academic Context
Data Science Project — St. John College of Engineering and Technology (SLRTCE, 2026-27).
Developed by Sourabh Patel, Chirag Khaire, and Khubaib Khan.
Under the guidance of Asst. Prof. Sheetal Dalvi.
