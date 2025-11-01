import argparse, json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def build_features(df: pd.DataFrame):
    df = df.copy()
    # Parse date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['state','county','date'])

    # Grouped new cases (cases is cumulative in most public datasets)
    df['cases'] = pd.to_numeric(df['cases'], errors='coerce')
    df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
    df[['cases','deaths']] = df[['cases','deaths']].fillna(0)

    df['new_cases'] = df.groupby(['state','county'])['cases'].diff().fillna(0)
    df['new_cases'] = df['new_cases'].clip(lower=0)

    # Lags & rolling stats
    for lag in [1,3,7]:
        df[f'new_cases_lag{lag}'] = df.groupby(['state','county'])['new_cases'].shift(lag)

    df['new_cases_roll7_mean'] = df.groupby(['state','county'])['new_cases'] \
                                     .transform(lambda s: s.rolling(7, min_periods=1).mean())
    df['dow'] = df['date'].dt.weekday  # 0=Mon

    # Label: did cases increase next day?
    df['new_cases_tomorrow'] = df.groupby(['state','county'])['new_cases'].shift(-1)
    df['label_increase'] = (df['new_cases_tomorrow'] > 0).astype(int)

    # Drop rows without full features/label
    feature_cols = ['new_cases','new_cases_lag1','new_cases_lag3','new_cases_lag7',
                    'new_cases_roll7_mean','deaths','dow','state','county']
    df = df.dropna(subset=feature_cols + ['label_increase'])

    # Encode state/county (tree model works fine with label encoding)
    state_le = LabelEncoder()
    county_le = LabelEncoder()
    df['state_le'] = state_le.fit_transform(df['state'].astype(str))
    df['county_le'] = county_le.fit_transform(df['county'].astype(str))

    X = df[['new_cases','new_cases_lag1','new_cases_lag3','new_cases_lag7',
            'new_cases_roll7_mean','deaths','dow','state_le','county_le']].values
    y = df['label_increase'].values

    meta = {
        'feature_names': ['new_cases','new_cases_lag1','new_cases_lag3','new_cases_lag7',
                          'new_cases_roll7_mean','deaths','dow','state_le','county_le'],
        'label_name': 'label_increase'
    }
    encoders = {'state_le': state_le, 'county_le': county_le}
    return X, y, meta, encoders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/us-covid_19-202.parquet",
                    help="Path to parquet/csv (default: dataset/us-covid_19-202.parquet)")
    ap.add_argument("--model-out", default="models/model.joblib")
    ap.add_argument("--meta-out", default="models/model_meta.json")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"[INFO] Reading {data_path} ...")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    X, y, meta, encoders = build_features(df)
    print(f"[INFO] Training rows: {X.shape[0]}, features: {X.shape[1]}")

    if X.shape[0] < 1000:
        print("[WARN] Few training rows; results may be poor.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    print("[INFO] Training RandomForestClassifier ...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[OK] Accuracy: {acc:.4f}")
    try:
        print(classification_report(y_test, y_pred))
    except Exception:
        pass

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'encoders': encoders}, args.model_out)

    meta_out = {
        'data_path': str(data_path),
        'metrics': {'accuracy': float(acc)},
        'features': meta['feature_names'],
        'label': meta['label_name'],
        'encoders': ['state_le','county_le']
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[SAVE] {args.model_out}")
    print(f"[SAVE] {args.meta_out}")
    print("[DONE] Training complete ✅")

if __name__ == "__main__":
    main()
