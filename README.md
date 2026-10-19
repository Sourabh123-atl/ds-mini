# COVID/Influenza Trend Analysis Using Social Media Data

Data Science mini-project (SLRTCE, 2026-27). Trend analysis of COVID-19 tweets with an India / Maharashtra / Mumbai focus, compared against official case counts, plus a sentiment model and an interactive Streamlit dashboard.

## Data (all public, no login needed)
| File (data/raw) | Source | Rows |
|---|---|---|
| covid19_tweets.csv | G. Preda, COVID19 Tweets (Kaggle `gpreda/covid19-tweets`), mirror https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv | 179,108 |
| corona_nlp.parquet | Coronavirus tweets NLP (Kaggle `datatattle/covid-19-nlp-text-classification`), mirror https://huggingface.co/datasets/argilla/twitter-coronavirus | 44,955 (41,157 train / 3,798 test) |
| case_time_series.csv, states.csv, districts.csv | covid19india.org archive, https://data.covid19india.org/csv/latest/ | 641 / 21,675 / 358,633 |
| full_data.csv | Our World in Data, https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/cases_deaths/full_data.csv | World series: 1,674 |

Raw files are not committed (~119 MB). Download them with:
```bash
python scripts/download_data.py
```
or manually:
```bash
cd data/raw
curl -LO https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv
curl -L -o corona_nlp.parquet https://huggingface.co/api/datasets/argilla/twitter-coronavirus/parquet/default/train/0.parquet
for f in case_time_series states districts; do curl -LO https://data.covid19india.org/csv/latest/$f.csv; done
curl -LO https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/cases_deaths/full_data.csv
```

## Install & run
The precomputed `data/processed/`, `models/`, screenshots and report are committed, so after installing you can run the dashboard straight away (`streamlit run app/streamlit_app.py`). To rebuild everything from raw data:
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/analyze.py        # ~20 s -> data/processed/*
.venv/bin/python scripts/train_model.py    # ~25 s -> models/sentiment_tfidf_lr.joblib, models/metrics.json
.venv/bin/streamlit run app/streamlit_app.py   # http://localhost:8501
```
`train_model.py` must run after `analyze.py` (it adds the model's predictions to `tweets.parquet`).

## Demo script
1. **Trends** – tweets/day vs new cases. Switch Region to *Mumbai* in the sidebar (2,488 tweets) and the case line switches to Mumbai.
2. **Sentiment** – daily mix; toggle *Sentiment source* between VADER and the trained model.
3. **Topics** – 8 NMF topics, share by week.
4. **Keywords & Hashtags** – type `vaccine` or `#lockdown` in the search box; compare India vs rest of world.
5. **Cases correlation** – lag correlation −7…+7 days.
6. **Try the model** – type any tweet, e.g. "Hospitals in Mumbai are running out of oxygen beds".
7. **Data** – filtered rows + CSV download.

## Model
TF-IDF (1–2 grams) + Logistic Regression, 3-class sentiment, official test split: accuracy 0.782, macro F1 0.759 (see `models/metrics.json`).

## Report & screenshots
- Screenshots: `artifacts/screenshots/` (regenerate with `.venv/bin/python scripts/screenshots.py` while the app runs on port 8517).
- Report: `report/DS_Mini_Project_Report.pdf` (build: `.venv/bin/python scripts-report/build_report.py && libreoffice --headless --convert-to pdf --outdir report report/DS_Mini_Project_Report.docx`). Guide name / team are constants at the top of `scripts-report/build_report.py`.

## Caveats
Tweets span only 25 days (25 Jul – 30 Aug 2020) and the collector capped most days at ~7,500 tweets, so raw volume reflects collection, not interest. Region flags come from free-text `user_location` ("Mumbai" includes Thane / Navi Mumbai). VADER reproduces the labelled corpus's labels with 0.935 accuracy, so the labels look lexicon-like.
