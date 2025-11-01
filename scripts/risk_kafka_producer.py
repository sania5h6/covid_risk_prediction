import argparse, json, time
from datetime import datetime
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer

def json_serializer(v): return json.dumps(v).encode("utf-8")

def main():
    ap = argparse.ArgumentParser(description="Pandas → Kafka producer")
    ap.add_argument("--data", default="dataset/us-covid_19-202.parquet",
                    help="Path to parquet/csv")
    ap.add_argument("--broker", required=True, help="Kafka bootstrap server, e.g. localhost:9092")
    ap.add_argument("--topic", required=True, help="Kafka topic name")
    ap.add_argument("--sleep", type=float, default=0.001, help="Sleep between messages (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="Send at most N rows (0 = all)")
    args = ap.parse_args()

    pth = Path(args.data)
    if not pth.exists():
        raise FileNotFoundError(f"Data file not found: {pth}")

    print(f"[INFO] Reading {pth} ...")
    df = pd.read_parquet(pth) if pth.suffix == ".parquet" else pd.read_csv(pth)

    # Ensure date & derive month/day
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day

    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Connecting to Kafka at {args.broker} ...")
    producer = KafkaProducer(bootstrap_servers=args.broker,
                             value_serializer=json_serializer,
                             acks="all", linger_ms=50)

    sent = 0
    for _, row in df.iterrows():
        payload = {
            "state":  (None if pd.isna(row.get("state"))  else str(row.get("state"))),
            "county": (None if pd.isna(row.get("county")) else str(row.get("county"))),
            "date":   (row.get("date").strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None),
            "cases":  (None if pd.isna(row.get("cases"))  else int(row.get("cases"))),
            "deaths": (None if pd.isna(row.get("deaths")) else int(row.get("deaths"))),
            "month":  (None if "month" not in row or pd.isna(row.get("month")) else int(row.get("month"))),
            "day":    (None if "day"   not in row or pd.isna(row.get("day"))   else int(row.get("day"))),
            "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        key = (payload["county"] or "unknown").encode()
        producer.send(args.topic, value=payload, key=key)

        sent += 1
        if args.limit and sent >= args.limit: break
        if args.sleep: time.sleep(args.sleep)
        if sent % 5000 == 0: print(f"[INFO] Sent {sent} records...")

    producer.flush(); producer.close()
    print(f"[OK] Finished sending {sent} messages to '{args.topic}'.")
if __name__ == "__main__":
    main()
