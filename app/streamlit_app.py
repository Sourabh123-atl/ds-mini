"""COVID/Influenza Trend Analysis Using Social Media Data — Streamlit dashboard.
Run: .venv/bin/streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
st.set_page_config(page_title="COVID Social Trends", page_icon="🦠", layout="wide")
SENT_COLORS = {"Negative": "#d62728", "Neutral": "#7f7f7f", "Positive": "#2ca02c"}
TERMS = ["fever", "cough", "vaccine", "oxygen", "lockdown", "flu/influenza", "mask", "hospital", "test", "death"]


@st.cache_data
def load():
    t = pd.read_parquet(P / "tweets.parquet")
    cases = pd.read_csv(P / "cases_daily.csv", parse_dates=["day"])
    lag = pd.read_csv(P / "lag_corr.csv")
    summ = json.loads((P / "summary.json").read_text())
    return t, cases, lag, summ


@st.cache_resource
def model():
    f = ROOT / "models/sentiment_tfidf_lr.joblib"
    return joblib.load(f) if f.exists() else None


t, cases, lag, summ = load()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
dmin, dmax = t.day.min().date(), t.day.max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
region = st.sidebar.radio("Region (from user_location)", ["Global", "India", "Maharashtra", "Mumbai"])
kw = st.sidebar.text_input("Keyword / #hashtag search", "", placeholder="e.g. vaccine, #lockdown")
sent_src = st.sidebar.radio("Sentiment source", ["VADER", "TF-IDF+LR model"], horizontal=True)
scol = "sentiment" if sent_src == "VADER" else "model_sentiment"
sents = st.sidebar.multiselect("Sentiment", ["Negative", "Neutral", "Positive"], ["Negative", "Neutral", "Positive"])

f = t
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    f = f[(f.day.dt.date >= dr[0]) & (f.day.dt.date <= dr[1])]
f = {"Global": f, "India": f[f.is_india], "Maharashtra": f[f.is_maha], "Mumbai": f[f.is_mumbai]}[region]
if kw.strip():
    k = kw.strip().lower()
    f = f[f.tags.str.contains(rf"\b{k[1:]}\b", regex=True)] if k.startswith("#") else f[f.text.str.lower().str.contains(k, regex=False)]
f = f[f[scol].isin(sents)]

st.title("🦠 COVID / Influenza Trend Analysis Using Social Media Data")
st.caption(f"COVID19 Tweets (Preda, Kaggle) · {summ['clean_rows']:,} tweets · {summ['date_min']} → {summ['date_max']} "
           f"· cases: covid19india.org archive + OWID")
c = st.columns(5)
c[0].metric("Tweets (filtered)", f"{len(f):,}")
c[1].metric("Days covered", f.day.nunique())
c[2].metric("Mean VADER compound", f"{f.compound.mean():+.3f}" if len(f) else "–")
c[3].metric("% Negative", f"{100*(f[scol]=='Negative').mean():.1f}%" if len(f) else "–")
c[4].metric("India share", f"{100*f.is_india.mean():.1f}%" if len(f) else "–")
if f.empty:
    st.warning("No tweets match these filters."); st.stop()

tabs = st.tabs(["📈 Trends", "😊 Sentiment", "🧩 Topics", "🔑 Keywords & Hashtags", "📊 Cases correlation", "🤖 Try the model", "🗂 Data"])

with tabs[0]:
    daily = f.groupby("day").size().rename("tweets").reset_index()
    ccol = {"Global": "world_cases_owid", "India": "india_cases", "Maharashtra": "maha_cases", "Mumbai": "mumbai_cases"}[region]
    m = daily.merge(cases[["day", ccol]], on="day", how="left")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=m.day, y=m.tweets, name="Tweets/day", marker_color="#1f77b4", opacity=.75)
    fig.add_scatter(x=m.day, y=m[ccol], name=f"New cases ({region}{', OWID 7-day avg' if region == 'Global' else ''})", mode="lines+markers", line_color="#d62728", secondary_y=True)
    fig.update_layout(title=f"Daily tweet volume vs new COVID cases — {region}", height=430, legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="tweets", secondary_y=False); fig.update_yaxes(title_text="new cases", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Note: the collector capped most days at ~7,500 tweets and some days are missing, so absolute volume reflects collection, not public interest. Use region share and per-tweet rates for comparisons.")
    a, b = st.columns(2)
    wk = f.assign(week=f.day.dt.to_period("W").dt.start_time).groupby(["week", "region"]).size().rename("tweets").reset_index()
    a.plotly_chart(px.bar(wk, x="week", y="tweets", color="region", title="Weekly tweets by region", height=380), use_container_width=True)
    share = t.groupby("day").agg(india=("is_india", "mean"), maha=("is_maha", "mean"), mumbai=("is_mumbai", "mean")).mul(100).reset_index()
    b.plotly_chart(px.line(share, x="day", y=["india", "maha", "mumbai"], markers=True, title="Share of all tweets from India / Maharashtra / Mumbai (%)", height=380), use_container_width=True)

with tabs[1]:
    sd = f.groupby(["day", scol]).size().rename("n").reset_index()
    sd["pct"] = 100 * sd.n / sd.groupby("day").n.transform("sum")
    st.plotly_chart(px.area(sd, x="day", y="pct", color=scol, color_discrete_map=SENT_COLORS, title=f"Daily sentiment mix ({sent_src}) — {region}", height=400), use_container_width=True)
    a, b = st.columns(2)
    reg = t.assign(R=np.select([t.is_mumbai, t.is_maha, t.is_india], ["Mumbai", "Maharashtra", "India"], "Rest of world"))
    rs = reg.groupby(["R", scol]).size().rename("n").reset_index(); rs["pct"] = 100 * rs.n / rs.groupby("R").n.transform("sum")
    a.plotly_chart(px.bar(rs, x="R", y="pct", color=scol, color_discrete_map=SENT_COLORS, title="Sentiment by region (all dates)", height=380), use_container_width=True)
    mc = f.groupby("day").compound.mean().rolling(3, min_periods=1).mean().reset_index()
    b.plotly_chart(px.line(mc, x="day", y="compound", markers=True, title="Mean VADER compound (3-day rolling)", height=380), use_container_width=True)
    st.subheader("Most negative / positive tweets in selection")
    x1, x2 = st.columns(2)
    x1.dataframe(f.nsmallest(8, "compound")[["day", "compound", "text"]], hide_index=True, use_container_width=True)
    x2.dataframe(f.nlargest(8, "compound")[["day", "compound", "text"]], hide_index=True, use_container_width=True)

with tabs[2]:
    tp = f.assign(week=f.day.dt.to_period("W").dt.start_time).groupby(["week", "topic_label"]).size().rename("n").reset_index()
    tp["share"] = 100 * tp.n / tp.groupby("week").n.transform("sum")
    st.plotly_chart(px.bar(tp, x="week", y="share", color="topic_label", title="NMF topic share by week (%)", height=430), use_container_width=True)
    a, b = st.columns([1, 1])
    a.plotly_chart(px.pie(f, names="topic_label", title="Topic share in selection", height=380, hole=.4), use_container_width=True)
    b.dataframe(pd.DataFrame([{"topic": d["label"].split(":")[0], "top 10 terms": ", ".join(d["top_terms"])} for d in summ["topics"]]), hide_index=True, use_container_width=True)

with tabs[3]:
    a, b = st.columns(2)
    tags = f.tags.str.split().explode().dropna()
    tags = tags[~tags.isin(["covid19", "coronavirus"])]
    top = tags.value_counts().head(20).rename_axis("hashtag").reset_index(name="count")
    a.plotly_chart(px.bar(top[::-1], x="count", y="hashtag", orientation="h", title="Top hashtags (excluding #covid19, #coronavirus)", height=520), use_container_width=True)
    tm = pd.DataFrame({"term": TERMS, "pct": [100 * f["t_" + k].mean() for k in TERMS]}).sort_values("pct")
    b.plotly_chart(px.bar(tm, x="pct", y="term", orientation="h", title="Symptom / health term mentions (% of tweets)", height=520), use_container_width=True)
    sel = st.multiselect("Terms over time", TERMS, ["vaccine", "lockdown", "mask", "flu/influenza"])
    if sel:
        tt = f.groupby("day")[["t_" + k for k in sel]].mean().mul(100).reset_index().rename(columns={"t_" + k: k for k in sel})
        st.plotly_chart(px.line(tt, x="day", y=sel, markers=True, title="Daily % of tweets mentioning each term", height=400), use_container_width=True)
    comp = pd.DataFrame({"term": TERMS, "India": [summ["term_pct_india"][k] for k in TERMS], "Rest of world": [summ["term_pct_rest"][k] for k in TERMS]})
    st.plotly_chart(px.bar(comp.melt("term", var_name="group", value_name="pct"), x="term", y="pct", color="group", barmode="group", title="India vs rest of world: term mention rate (%)", height=380), use_container_width=True)

with tabs[4]:
    st.markdown("**Lag correlation:** Pearson r between daily tweets on day *d* and new cases on day *d + lag* (25 collection days).")
    st.plotly_chart(px.line(lag, x="lag_days", y="pearson_r", color="series", markers=True, title="Lag correlation: tweet volume vs new cases", height=420).add_hline(y=0, line_dash="dot"), use_container_width=True)
    dl = pd.read_csv(P / "daily.csv", parse_dates=["day"])
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(dl, x="india_cases", y="india", trendline="ols" if False else None, title="India tweets/day vs India new cases/day", hover_data=["day"], height=380), use_container_width=True)
    cc = cases[(cases.day >= "2020-03-01") & (cases.day <= "2021-06-30")]
    fig = px.line(cc, x="day", y=["india_cases", "maha_cases", "mumbai_cases"], title="Daily new cases: India / MH / Mumbai", height=380)
    fig.add_vrect(x0=summ["date_min"], x1=summ["date_max"], fillcolor="orange", opacity=.2, annotation_text="tweet window")
    b.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(summ["best_lag"]).T, use_container_width=True)

with tabs[5]:
    mdl = model()
    if mdl is None:
        st.warning("Model not found. Run scripts/train_model.py")
    else:
        met = json.loads((ROOT / "models/metrics.json").read_text())
        c = st.columns(3)
        c[0].metric("Test accuracy", f"{met['accuracy']:.3f}"); c[1].metric("Macro F1", f"{met['f1_macro']:.3f}"); c[2].metric("Test tweets", f"{met['test_size']:,}")
        txt = st.text_area("Type a COVID/flu tweet", "Hospitals in Mumbai are running out of oxygen beds, this is terrifying")
        if txt.strip():
            pr = mdl.predict_proba([txt])[0]
            lab = mdl.classes_[pr.argmax()]
            st.success(f"Predicted sentiment: **{lab}** ({pr.max():.1%})")
            st.plotly_chart(px.bar(x=mdl.classes_, y=pr, color=mdl.classes_, color_discrete_map=SENT_COLORS, labels={"x": "class", "y": "probability"}, height=320), use_container_width=True)
        cm = np.array(met["confusion_matrix"])
        st.plotly_chart(px.imshow(cm, x=met["confusion_matrix_labels"], y=met["confusion_matrix_labels"], text_auto=True, color_continuous_scale="Blues", labels=dict(x="predicted", y="true"), title="Confusion matrix (official test split)", height=380), use_container_width=True)

with tabs[6]:
    st.dataframe(f[["date", "region", "user_location", "sentiment", "model_sentiment", "compound", "topic_label", "tags", "text"]].head(500), hide_index=True, use_container_width=True)
    st.download_button("Download filtered CSV", f.drop(columns=[c for c in f.columns if c.startswith("t_")]).to_csv(index=False), "filtered_tweets.csv")
