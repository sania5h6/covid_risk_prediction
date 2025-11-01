# COVID-19 Risk Predictions

End-to-end dashboard that loads US COVID-19 data into PostgreSQL (AWS RDS) and serves a Streamlit app for exploration and simple model signals.

## Features
- Streamlit dashboard (`scripts/streamlit_dashboard.py`)
- PostgreSQL (AWS RDS) backend: `covid_events` table
- Loader scripts (CSV/Parquet → RDS)
- Optional Big Data hooks (Kafka, Spark/HDFS) for future streaming

## Quick Start

```bash
git clone git@github.com:sania5h6/covid_risk_prediction.git
cd covid_risk_prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your RDS endpoint & credentials

# Load data (Parquet fast path)
python scripts/load_wide_parquet_to_rds.py

# Run dashboard locally
streamlit run scripts/streamlit_dashboard.py

nano README.md









OF
