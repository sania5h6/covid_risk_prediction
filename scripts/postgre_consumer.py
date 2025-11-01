import argparse
import json
import time
from kafka import KafkaConsumer
import psycopg2
from psycopg2.extras import execute_values

DDL = """
CREATE TABLE IF NOT EXISTS covid_events (
  id SERIAL PRIMARY KEY,
  state TEXT,
  county TEXT,
  date DATE,
  cases INTEGER,
  deaths INTEGER,
  month INTEGER,
  day INTEGER,
  ingested_at TIMESTAMPTZ
);
"""

INSERT_SQL = """
INSERT INTO covid_events (state, county, date, cases, deaths, month, day, ingested_at)
VALUES %s
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--pg-user", required=True)
    ap.add_argument("--pg-password", required=True)
    ap.add_argument("--pg-db", required=True)
    ap.add_argument("--pg-host", default="localhost")
    ap.add_argument("--pg-port", default=5432, type=int)
    ap.add_argument("--batch", default=1000, type=int, help="Batch size")
    args = ap.parse_args()

    # Connect to Postgres
    print(f"[INFO] Connecting to Postgres at {args.pg_host}:{args.pg_port} ...")
    conn = psycopg2.connect(
        dbname=args.pg_db, user=args.pg_user, password=args.pg_password,
        host=args.pg_host, port=args.pg_port
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(DDL)
    print("[INFO] Ensured table covid_events exists.")

    # Kafka consumer
    print(f"[INFO] Connecting to Kafka at {args.broker} ...")
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="covid_consumer_group",
        consumer_timeout_ms=300000
    )

    print(f"[INFO] Consuming messages from topic '{args.topic}' ...")
    batch = []
    total = 0
    try:
        for msg in consumer:
            v = msg.value
            row = (
                v.get("state"),
                v.get("county"),
                v.get("date"),
                v.get("cases"),
                v.get("deaths"),
                v.get("month"),
                v.get("day"),
                v.get("ingested_at"),
            )
            batch.append(row)
            if len(batch) >= args.batch:
                with conn.cursor() as cur:
                    execute_values(cur, INSERT_SQL, batch)
                total += len(batch)
                print(f"[INFO] Inserted {total} rows")
                batch.clear()
        if batch:
            with conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, batch)
            total += len(batch)
            print(f"[INFO] Inserted final batch. Total {total} rows")
    finally:
        consumer.close()
        conn.close()
        print("[DONE] Consumer stopped.")
if __name__ == "__main__":
    main()
