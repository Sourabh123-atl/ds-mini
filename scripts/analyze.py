#!/usr/bin/env python3
"""Pipeline: clean tweets, flag regions, sentiment, hashtags, terms, NMF topics, case join, lag correlation.
Run: .venv/bin/python scripts/analyze.py
"""
import ast, json, re, warnings
warnings.filterwarnings("ignore", "This pattern")
from pathlib import Path
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

MUMBAI = r"\b(mumbai|bombay|navi mumbai|thane|andheri|bandra)\b"
MAHA = r"\b(maharashtra|pune|nagpur|nashik|aurangabad|kolhapur|solapur)\b|" + MUMBAI
INDIA = (r"\b(india|bharat|hindustan|new delhi|delhi|bengaluru|bangalore|chennai|kolkata|hyderabad|ahmedabad|"
         r"jaipur|lucknow|kerala|kochi|tamil nadu|karnataka|gujarat|rajasthan|uttar pradesh|bihar|odisha|"
         r"west bengal|telangana|andhra pradesh|punjab|haryana|assam|goa|noida|gurgaon|gurugram|indore|bhopal|"
         r"patna|chandigarh|surat|vadodara|kanpur|madhya pradesh|jharkhand|uttarakhand|kashmir)\b|" + MAHA)
TERMS = {
    "fever": r"\bfever", "cough": r"\bcough", "vaccine": r"\bvaccin", "oxygen": r"\boxygen",
    "lockdown": r"\block ?down", "flu/influenza": r"\b(flu|influenza)\b", "mask": r"\bmasks?\b",
    "hospital": r"\bhospital", "test": r"\btest(s|ing|ed)?\b", "death": r"\b(death|deaths|died|dead)\b",
}

def main():
    t = pd.read_csv(RAW / "covid19_tweets.csv")
    n_raw = len(t)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date", "text"]).drop_duplicates(subset=["text", "user_name"]).copy()
    t["day"] = t["date"].dt.normalize()
    t = t[t["day"] >= "2020-07-25"]  # 24 Jul has only a few hundred late-night tweets
    loc = t["user_location"].fillna("").str.lower()
    t["is_mumbai"] = loc.str.contains(MUMBAI, regex=True)
    t["is_maha"] = loc.str.contains(MAHA, regex=True)
    t["is_india"] = loc.str.contains(INDIA, regex=True)
    t["region"] = np.where(t.is_mumbai, "Mumbai", np.where(t.is_maha, "Maharashtra (other)", np.where(t.is_india, "India (other)", "Rest of world / unknown")))

    sia = SentimentIntensityAnalyzer()
    t["compound"] = [sia.polarity_scores(x)["compound"] for x in t["text"]]
    t["sentiment"] = pd.cut(t["compound"], [-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"]).astype(str)

    def tags(row):
        s = row if isinstance(row, str) else ""
        try:
            v = ast.literal_eval(s) if s.startswith("[") else []
        except Exception:
            v = []
        return [x.lower() for x in v]
    t["tags"] = t["hashtags"].map(tags)
    low = t["text"].str.lower()
    for k, rx in TERMS.items():
        t["t_" + k] = low.str.contains(rx, regex=True)

    # NMF topics
    clean = low.str.replace(r"https?://\S+|@\w+|&amp;", " ", regex=True).str.replace(r"[^a-z# ]", " ", regex=True)
    extra = ["covid", "covid19", "coronavirus", "corona", "amp", "just", "like", "new", "today", "people", "says", "said", "don", "via", "covid_19", "pandemic", "s", "t"]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    vec = TfidfVectorizer(max_features=6000, min_df=20, max_df=0.4, stop_words=list(ENGLISH_STOP_WORDS.union(extra)), token_pattern=r"(?u)\b[a-z][a-z]{2,}\b")
    X = vec.fit_transform(clean)
    K = 8
    nmf = NMF(n_components=K, random_state=42, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for k in range(K):
        top = terms[np.argsort(nmf.components_[k])[::-1][:10]].tolist()
        topics.append({"topic": k, "top_terms": top, "label": f"T{k}: " + ", ".join(top[:3])})
    t["topic"] = np.where(W.max(1) > 0, W.argmax(1), -1)
    lab = {d["topic"]: d["label"] for d in topics}; lab[-1] = "Unassigned"
    t["topic_label"] = t["topic"].map(lab)

    keep = ["date", "day", "user_location", "user_verified", "user_followers", "text", "source", "region",
            "is_india", "is_maha", "is_mumbai", "compound", "sentiment", "tags", "topic", "topic_label"] + ["t_" + k for k in TERMS]
    slim = t[keep].copy()
    slim["tags"] = slim["tags"].map(lambda v: " ".join(v))
    slim.to_parquet(OUT / "tweets.parquet", index=False)

    # Cases
    ind = pd.read_csv(RAW / "case_time_series.csv", parse_dates=["Date_YMD"]).rename(columns={"Date_YMD": "day", "Daily Confirmed": "india_cases"})[["day", "india_cases"]]
    st = pd.read_csv(RAW / "states.csv", parse_dates=["Date"])
    mh = st[st.State == "Maharashtra"].sort_values("Date")
    mh = pd.DataFrame({"day": mh.Date, "maha_cases": mh.Confirmed.diff()})
    ds = pd.read_csv(RAW / "districts.csv", parse_dates=["Date"])
    mu = ds[(ds.State == "Maharashtra") & (ds.District == "Mumbai")].sort_values("Date")
    mu = pd.DataFrame({"day": mu.Date, "mumbai_cases": mu.Confirmed.diff()})
    ow = pd.read_csv(RAW / "full_data.csv", parse_dates=["date"])
    ow = ow[ow.location == "World"].rename(columns={"date": "day"})
    # OWID World new_cases is reported in weekly batches; use the 7-day sum / 7 as a daily average
    ow["world_cases_owid"] = ow["weekly_cases"] / 7
    ow = ow[["day", "world_cases_owid"]]
    cases = ind.merge(mh, on="day", how="outer").merge(mu, on="day", how="outer").merge(ow, on="day", how="outer").sort_values("day")
    cases = cases[(cases.day >= "2020-03-01") & (cases.day <= "2021-10-31")]
    cases.to_csv(OUT / "cases_daily.csv", index=False)

    # Daily aggregates
    g = t.groupby("day")
    daily = pd.DataFrame({
        "tweets": g.size(), "india": g.is_india.sum(), "maha": g.is_maha.sum(), "mumbai": g.is_mumbai.sum(),
        "mean_compound": g.compound.mean(),
        "india_compound": t[t.is_india].groupby("day").compound.mean(),
    }).reset_index()
    daily["india_share_pct"] = 100 * daily.india / daily.tweets
    daily = daily.merge(cases, on="day", how="left")
    daily.to_csv(OUT / "daily.csv", index=False)

    # Lag correlation on India share vs India cases (lag L: tweets on day d vs cases on day d+L)
    cs = cases.set_index("day")
    lags = []
    for L in range(-7, 8):
        shifted = daily["day"] + pd.Timedelta(days=L)
        for col, cname in [("india", "india_cases"), ("india_share_pct", "india_cases"), ("maha", "maha_cases"), ("mumbai", "mumbai_cases")]:
            y = cs[cname].reindex(shifted).values
            x = daily[col].values
            m = ~np.isnan(y)
            r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 3 else np.nan
            lags.append({"lag_days": L, "series": f"{col} vs {cname}", "pearson_r": r, "n": int(m.sum())})
    pd.DataFrame(lags).to_csv(OUT / "lag_corr.csv", index=False)

    summary = {
        "raw_rows": n_raw, "clean_rows": len(t), "days": int(t.day.nunique()),
        "date_min": str(t.day.min().date()), "date_max": str(t.day.max().date()),
        "india_rows": int(t.is_india.sum()), "maha_rows": int(t.is_maha.sum()), "mumbai_rows": int(t.is_mumbai.sum()),
        "sentiment_share": t.sentiment.value_counts(normalize=True).round(4).to_dict(),
        "sentiment_share_india": t[t.is_india].sentiment.value_counts(normalize=True).round(4).to_dict(),
        "sentiment_share_mumbai": t[t.is_mumbai].sentiment.value_counts(normalize=True).round(4).to_dict(),
        "mean_compound": round(float(t.compound.mean()), 4), "mean_compound_india": round(float(t[t.is_india].compound.mean()), 4),
        "mean_compound_mumbai": round(float(t[t.is_mumbai].compound.mean()), 4),
        "term_counts": {k: int(t["t_" + k].sum()) for k in TERMS},
        "term_pct_india": {k: round(100 * float(t[t.is_india]["t_" + k].mean()), 2) for k in TERMS},
        "term_pct_rest": {k: round(100 * float(t[~t.is_india]["t_" + k].mean()), 2) for k in TERMS},
        "top_hashtags": t.tags.explode().value_counts().head(20).to_dict(),
        "top_hashtags_india": t[t.is_india].tags.explode().value_counts().head(15).to_dict(),
        "topics": topics, "topic_share": t.topic_label.value_counts(normalize=True).round(4).to_dict(),
        "n_cases_rows": {"case_time_series": len(ind), "states": len(st), "districts": len(ds), "owid_full_data": len(ow)},
    }
    lg = pd.DataFrame(lags)
    for s in lg.series.unique():
        sub = lg[lg.series == s].dropna()
        b = sub.loc[sub.pearson_r.abs().idxmax()]
        summary.setdefault("best_lag", {})[s] = {"lag_days": int(b.lag_days), "r": round(float(b.pearson_r), 3), "r_lag0": round(float(sub[sub.lag_days == 0].pearson_r.iloc[0]), 3), "n": int(b.n)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "topics.json").write_text(json.dumps(topics, indent=2))
    print(json.dumps(summary, indent=1, default=str))

if __name__ == "__main__":
    main()
