#!/usr/bin/env python3
"""Download the raw datasets into data/raw (public URLs, no login).
Run: python scripts/download_data.py
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "covid19_tweets.csv": "https://raw.githubusercontent.com/gabrielpreda/covid-19-tweets/master/covid19_tweets.csv",
    "corona_nlp.parquet": "https://huggingface.co/api/datasets/argilla/twitter-coronavirus/parquet/default/train/0.parquet",
    "case_time_series.csv": "https://data.covid19india.org/csv/latest/case_time_series.csv",
    "states.csv": "https://data.covid19india.org/csv/latest/states.csv",
    "districts.csv": "https://data.covid19india.org/csv/latest/districts.csv",
    "full_data.csv": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/cases_deaths/full_data.csv",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        if dst.exists() and dst.stat().st_size > 0:
            print("exists", name); continue
        print("downloading", name)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print("  ", dst.stat().st_size, "bytes")
