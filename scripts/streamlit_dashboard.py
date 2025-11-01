# scripts/streamlit_dashboard.py
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import us
import joblib

# ---------- App config ----------
st.set_page_config(page_title="COVID-19 Risk Predictions", layout="wide")
st.title("COVID-19 Risk Predictions Dashboard")
st.markdown("Live view of ingested COVID-19 data and model signals.")

# ---------- .env & DB ----------
load_dotenv()  # reads .env at project root
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "covid")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "covidpw")
PG_DB = os.getenv("POSTGRES_DB", "covid_db")

ENGINE = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

STATE_CODES = {s.name: s.abbr for s in us.states.STATES}

@st.cache_data(show_spinner=False)
def load_events(limit: int | None = 200_000) -> pd.DataFrame:
    sql = """
      SELECT state, county, date::date AS date, cases, deaths, month, day, ingested_at
      FROM covid_events
    """
    if limit:
        sql = f"SELECT state, county, date::date AS date, cases, deaths, month, day, ingested_at FROM covid_events LIMIT {int(limit)}"
    df = pd.read_sql(text(sql), ENGINE)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df["state_code"] = df["state"].map(STATE_CODES)
    df = df.dropna(subset=["date"]).sort_values(["state","county","date"])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["state","county","date"]).copy()
    df["new_cases"] = df.groupby(["state","county"])["cases"].diff().clip(lower=0).fillna(0)
    for lag in [1,3,7]:
        df[f"new_cases_lag{lag}"] = df.groupby(["state","county"])["new_cases"].shift(lag)
    df["new_cases_roll7_mean"] = df.groupby(["state","county"])["new_cases"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["dow"] = df["date"].dt.weekday
    return df

def load_model():
    p = Path("models/model.joblib")
    if p.exists():
        obj = joblib.load(p)
        return obj.get("model"), obj.get("encoders")
    return None, None

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Data")
    limit = st.slider("Max rows to load", 10_000, 250_000, 100_000, step=10_000)
    if st.button("Refresh data"):
        st.cache_data.clear()

# ---------- Load data ----------
with st.status("Loading data from PostgreSQL…", expanded=False):
    try:
        df = load_events(limit)
        st.write(f"Loaded **{len(df):,}** rows from `covid_events` @ {PG_HOST}:{PG_PORT}")
    except Exception as e:
        st.error(f"DB load error: {e}")
        st.stop()

if df.empty:
    st.warning("Database table `covid_events` is empty. Start producer & consumer, then refresh.")
    st.stop()

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("States", int(df["state"].nunique()))
c2.metric("Counties", int(df["county"].nunique()))
latest_dt = pd.to_datetime(df["date"].max()).date()
c3.metric("Latest Date", latest_dt.strftime("%Y-%m-%d"))   # <-- fixed: pass str, not date object
c4.metric("Total Rows", f"{len(df):,}")

# ---------- Trends ----------
st.subheader("Nationwide trends")
daily = df.groupby("date", as_index=False)[["cases","deaths"]].sum()
st.plotly_chart(px.line(daily, x="date", y="cases", title="Cumulative cases"), use_container_width=True)
st.plotly_chart(px.line(daily, x="date", y="deaths", title="Cumulative deaths"), use_container_width=True)

# ---------- Top states/counties ----------
st.subheader("Top states / counties (latest snapshot)")
latest = df[df["date"] == df["date"].max()].copy()
top_states = latest.groupby("state", as_index=False)[["cases","deaths"]].sum().sort_values("cases", ascending=False).head(15)
st.plotly_chart(px.bar(top_states, x="state", y="cases", title=f"Top 15 states by cases on {latest_dt}"), use_container_width=True)
top_counties = latest.sort_values("cases", ascending=False).head(15)
st.plotly_chart(px.bar(top_counties, x="county", y="cases", title=f"Top 15 counties by cases on {latest_dt}"), use_container_width=True)

# ---------- Map ----------
st.subheader("Map: cases by state (latest)")
state_agg = latest.groupby(["state","state_code"], as_index=False)["cases"].sum()
state_agg = state_agg.dropna(subset=["state_code"])
st.plotly_chart(
    px.choropleth(
        state_agg, locations="state_code", locationmode="USA-states",
        color="cases", scope="usa", color_continuous_scale="Reds",
        hover_data=["state"], title="Cases by state (latest)"
    ),
    use_container_width=True
)

# ---------- Model signal (optional) ----------
st.subheader("Model signal: probability of cases increasing tomorrow")
model, encoders = load_model()
if model is None:
    st.info("Model not found at `models/model.joblib`. Train it to see predictions.")
else:
    feats = build_features(df)
    last_per_county = feats.sort_values("date").groupby(["state","county"], as_index=False).tail(1)
    # Encode like training
    state_le = encoders["state_le"]; county_le = encoders["county_le"]
    last_per_county["state_le"]  = state_le.transform(last_per_county["state"].astype(str))
    last_per_county["county_le"] = county_le.transform(last_per_county["county"].astype(str))
    feature_cols = [
        "new_cases","new_cases_lag1","new_cases_lag3","new_cases_lag7",
        "new_cases_roll7_mean","deaths","dow","state_le","county_le"
    ]
    for c in feature_cols:
        if c not in last_per_county: last_per_county[c] = 0.0
    X = last_per_county[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    last_per_county["p_increase"] = proba
    st.dataframe(
        last_per_county[["state","county","date","new_cases","p_increase"]]
        .sort_values("p_increase", ascending=False).head(50),
        use_container_width=True
    )

# ---------- Raw table ----------
st.subheader("Raw table (sample)")
st.dataframe(df.sort_values(["state","county","date"]).head(5000), use_container_width=True)

