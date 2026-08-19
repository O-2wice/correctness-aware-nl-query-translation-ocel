"""Refresh saved benchmark and evaluation hashes without calling an LLM.

This script executes the checked-in gold SQL and saved predicted SQL against
the local OCEL parquet views, then rewrites:

- benchmark/nl2ocel_benchmark_{v1,dev,test}.csv gold_result_hash
- outputs/reports/baseline_*.csv / pipeline_*.csv pred_hash, gold_hash, den_acc

It uses nl2ocel.result_hash.hash_dataframe, so floating aggregate values are
canonicalized consistently across gold and predicted results.
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2ocel.result_hash import hash_dataframe

BENCHMARK = ROOT / "benchmark"
REPORTS = ROOT / "outputs" / "reports"

BENCHMARK_FILES = [
    BENCHMARK / "nl2ocel_benchmark_v1.csv",
    BENCHMARK / "nl2ocel_benchmark_dev.csv",
    BENCHMARK / "nl2ocel_benchmark_test.csv",
]

RESULT_FILES = [
    REPORTS / "baseline_b1_dev.csv",
    REPORTS / "baseline_b1_test.csv",
    REPORTS / "baseline_b2_dev.csv",
    REPORTS / "baseline_b2_test.csv",
    REPORTS / "baseline_b3_dev.csv",
    REPORTS / "baseline_b3_test.csv",
    REPORTS / "pipeline_dev.csv",
    REPORTS / "pipeline_test.csv",
]


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def open_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    ocel = str(ROOT / "data" / "processed" / "ocel").replace("\\", "/").replace("'", "''")
    for table in ("events", "objects", "relations"):
        conn.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{ocel}/{table}.parquet')"
        )
    return conn


def sql_hash(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, str]:
    try:
        return hash_dataframe(conn.execute(sql).df()), ""
    except Exception as exc:
        return "", str(exc).splitlines()[0][:300]


def refresh_benchmark(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    qid_to_hash: dict[str, str] = {}
    for path in BENCHMARK_FILES:
        df = pd.read_csv(path)
        errors = []
        for idx, row in df.iterrows():
            h, err = sql_hash(conn, str(row["gold_sql"]))
            if err:
                errors.append(f"{row['qid']}: {err}")
            else:
                df.at[idx, "gold_result_hash"] = h
                qid_to_hash[str(row["qid"])] = h
        write_csv_atomic(df, path)
        print(f"refreshed {path.relative_to(ROOT)} ({len(df)} rows)")
        if errors:
            raise RuntimeError(f"Gold SQL failures in {path.name}: " + "; ".join(errors[:5]))
    return qid_to_hash


def refresh_results(conn: duckdb.DuckDBPyConnection, qid_to_hash: dict[str, str]) -> None:
    for path in RESULT_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in ("gold_hash", "pred_hash", "pred_sql", "exec_error"):
            if col in df.columns:
                df[col] = df[col].astype("object").where(df[col].notna(), "")
        for idx, row in df.iterrows():
            qid = str(row.get("qid", ""))
            gold_hash = qid_to_hash.get(qid, str(row.get("gold_hash", "")))
            if "gold_hash" in df.columns:
                df.at[idx, "gold_hash"] = gold_hash

            pred_sql = row.get("pred_sql", "")
            pred_hash = ""
            exec_ok = 0
            exec_error = str(row.get("exec_error", "") or "")
            if isinstance(pred_sql, str) and pred_sql.strip():
                pred_hash, err = sql_hash(conn, pred_sql)
                exec_ok = int(not err)
                if err:
                    exec_error = err
                else:
                    exec_error = ""

            if "pred_hash" in df.columns:
                df.at[idx, "pred_hash"] = pred_hash
            if "exec_ok" in df.columns:
                if "status" in df.columns:
                    df.at[idx, "exec_ok"] = int(str(row.get("status", "")) == "accept")
                else:
                    df.at[idx, "exec_ok"] = exec_ok
            if "den_acc" in df.columns:
                df.at[idx, "den_acc"] = int(bool(pred_hash) and pred_hash == gold_hash)
            if "exec_error" in df.columns:
                df.at[idx, "exec_error"] = exec_error

        write_csv_atomic(df, path)
        print(f"refreshed {path.relative_to(ROOT)} ({len(df)} rows)")


def main() -> None:
    conn = open_conn()
    try:
        qid_to_hash = refresh_benchmark(conn)
        refresh_results(conn, qid_to_hash)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
