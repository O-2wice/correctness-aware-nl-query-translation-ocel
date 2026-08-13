"""
run_full_eval.py — evaluation runner

Runs all baselines (B1/B2/B3) plus Method M on the selected benchmark split.
Saves partial results after every question, so runs are safe to interrupt and
resume.

Usage (from project root, with venv):
    .venv/Scripts/python scripts/run_full_eval.py [--mode all|b1|b2|b3|method_m] [--split dev|test] [--resume]

Modes:
    all      — run everything sequentially (default)
    b1       — zero_shot (Rajkumar 2022 Create Table + Select 3)
    b2       — few_shot  (Nan 2023 Similarity-Diversity + augmentation + voting)
    b3       — din_sql   (Pourreza 2023 4-stage: slowest; ~4-7 LLM calls/question)
    method_m — constrained pipeline

Flags:
    --resume  — skip questions already saved in output CSV

Outputs:
    outputs/reports/nb03_b1_results.csv
    outputs/reports/nb03_b2_results.csv
    outputs/reports/nb03_b3_results.csv
    outputs/reports/nb04_method_m_dev.csv
    outputs/reports/gate_e_comparison.csv   (summary, after all dev runs)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from nl2ocel.llm_client import default_model_for_backend
from nl2ocel.result_hash import hash_dataframe


# ── Estimate remaining time ───────────────────────────────────────────────────

def eta_str(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    per = elapsed_s / done
    rem = per * (total - done)
    if rem < 60:
        return f"{rem:.0f}s"
    return f"{rem/60:.1f}min"


# ── Load existing partial results ─────────────────────────────────────────────

def load_done_qids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["qid"].tolist())


# ── Append one row to CSV ─────────────────────────────────────────────────────

def append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Baseline runner ───────────────────────────────────────────────────────────

def run_baseline(mode: str, dev_df: pd.DataFrame, out_path: Path, resume: bool,
                 backend: str = "deepseek", api_key: str | None = None) -> None:
    from nl2ocel.baseline_translator import BaselineTranslator, fetch_sample_rows_duckdb

    catalog_path   = ROOT / "configs" / "schema_catalog.json"
    whitelist_path = ROOT / "configs" / "relation_whitelist.json"
    ocel_dir       = ROOT / "data" / "processed" / "ocel"
    benchmark_path = ROOT / "benchmark" / "nl2ocel_benchmark_dev.csv"

    catalog   = json.loads(catalog_path.read_text(encoding="utf-8"))
    whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))["whitelist"]

    # Fetch 3 sample rows per OCEL table for Rajkumar 2022 'Select 3' prompt (B1).
    # Cheap one-off DuckDB reads — totals <1 MB.
    sample_rows = fetch_sample_rows_duckdb(ocel_dir, n=3)

    # Benchmark rows are needed by the few-shot baseline (B2) to build its
    # fixed-examples prompt. Load once; translator filters by qid internally.
    benchmark_rows = []
    if benchmark_path.exists():
        import csv as _csv
        with open(benchmark_path, encoding="utf-8") as f:
            benchmark_rows = list(_csv.DictReader(f))

    # Open the DuckDB connection FIRST so we can build an `executor` callable
    # before constructing the translator. Nan 2023's few-shot voting loop
    # needs to execute candidate SQL as it runs.
    import duckdb
    conn = duckdb.connect()
    _ocel = str(ocel_dir).replace("\\", "/").replace("'", "''")
    conn.execute(f"CREATE VIEW events    AS SELECT * FROM read_parquet('{_ocel}/events.parquet')")
    conn.execute(f"CREATE VIEW objects   AS SELECT * FROM read_parquet('{_ocel}/objects.parquet')")
    conn.execute(f"CREATE VIEW relations AS SELECT * FROM read_parquet('{_ocel}/relations.parquet')")

    def _sql_executor(sql: str) -> tuple[bool, str]:
        """Run candidate SQL and return (exec_ok, result_hash).

        Used by Nan 2023's voting loop to discard exec-error shots before
        the majority vote. Any exception is caught and surfaced as
        `(False, "")` — we never let a bad candidate crash the eval run.
        """
        if not sql or not sql.strip():
            return (False, "")
        try:
            df = conn.execute(sql).df()
            return (True, hash_dataframe(df))
        except Exception:
            return (False, "")

    model = default_model_for_backend(backend)
    translator = BaselineTranslator(
        mode=mode, catalog=catalog, whitelist=whitelist,
        benchmark_rows=benchmark_rows,
        sample_rows=sample_rows,
        sql_executor=_sql_executor,
        model=model, backend=backend, api_key=api_key,
    )

    # Exclude demo-seed questions from B2 (few_shot) evaluation: those QIDs
    # appear in the demonstration pool, so evaluating on them conflates
    # retrieval recall with in-context memorisation even with leave-one-out.
    if mode == "few_shot":
        from nl2ocel.baseline_translator import FEW_SHOT_QIDS
        dev_df = dev_df[~dev_df["qid"].isin(FEW_SHOT_QIDS)].reset_index(drop=True)

    done_qids = load_done_qids(out_path) if resume else set()
    total = len(dev_df)
    t_start = time.time()
    done_count = len(done_qids)

    print(f"\n{'='*60}")
    print(f"Mode: {mode}  |  Total: {total}  |  Resuming from: {done_count}")
    print(f"{'='*60}")

    for i, row in dev_df.iterrows():
        qid = row["qid"]
        if qid in done_qids:
            continue

        q = row["nl_question"]
        gold_hash = row.get("gold_result_hash", "")
        out = translator.translate(q, qid=qid)

        # Execute pred SQL
        exec_ok  = 0
        den_acc  = 0
        pred_hash = ""
        exec_err  = ""
        if out["pred_sql"] and not out["llm_error"]:
            try:
                df_res   = conn.execute(out["pred_sql"]).df()
                exec_ok  = 1
                pred_hash = hash_dataframe(df_res)
                den_acc   = int(pred_hash == gold_hash)
            except Exception as e:
                exec_err = str(e)[:200]

        result_row = {
            "qid":           qid,
            "nl_question":   q,
            "query_class":   row.get("query_class", ""),
            "difficulty":    row.get("difficulty", ""),
            "gold_hash":     gold_hash,
            "pred_sql":      out["pred_sql"] or "",
            "pred_hash":     pred_hash,
            "exec_ok":       exec_ok,
            "den_acc":       den_acc,
            "join_hall":     int(out.get("join_hall", False)),
            "llm_error":     int(bool(out.get("llm_error"))),
            "latency_s":     out.get("latency_s", 0),
            "exec_error":    exec_err,
        }
        append_row(out_path, result_row)
        done_count += 1
        elapsed = time.time() - t_start
        status = "OK" if den_acc else ("EXEC_FAIL" if not exec_ok else "WRONG")
        print(f"[{done_count:02d}/{total}] {qid} ({row.get('query_class','')}/{row.get('difficulty','')}) "
              f"-> {status}  {out.get('latency_s',0):.1f}s  ETA:{eta_str(elapsed, done_count, total)}")

    print(f"\nSaved: {out_path}")


# ── Method M runner ───────────────────────────────────────────────────────────

def run_method_m(dev_df: pd.DataFrame, out_path: Path, resume: bool,
                 backend: str = "deepseek", api_key: str | None = None) -> None:
    from nl2ocel.pipeline import ConstrainedPipeline

    model = default_model_for_backend(backend)
    pipeline = ConstrainedPipeline.from_project_root(ROOT, backend=backend,
                                                     model=model, api_key=api_key)

    done_qids = load_done_qids(out_path) if resume else set()
    total = len(dev_df)
    t_start = time.time()
    done_count = len(done_qids)

    print(f"\n{'='*60}")
    print(f"Mode: method_m  |  Total: {total}  |  Resuming from: {done_count}")
    print(f"{'='*60}")

    for _, row in dev_df.iterrows():
        qid = row["qid"]
        if qid in done_qids:
            continue

        q         = row["nl_question"]
        gold_hash = row.get("gold_result_hash", "")

        r = pipeline.run(q)
        hit = int(bool(r.result_hash) and r.result_hash == gold_hash)
        grounding = r.provenance.get("grounding", {}) if r.provenance else {}

        result_row = {
            "qid":           qid,
            "nl_question":   q,
            "query_class":   row.get("query_class", ""),
            "difficulty":    row.get("difficulty", ""),
            "gold_hash":     gold_hash,
            "pred_sql":      r.pred_sql or "",
            "pred_hash":     r.result_hash or "",
            "exec_ok":       int(r.status == "accept"),
            "den_acc":       hit,
            "ir_attempts":   r.ir_attempts,
            "verify_errors": "; ".join(r.verify_errors),
            "exec_error":    r.exec_error or "",
            "latency_s":     r.latency_s,
            "status":        r.status,
            "intent":        r.ir.get("intent", "") if r.ir else "",
            "joins_used":    json.dumps(r.provenance.get("joins_used", [])),
            "grounding_status": grounding.get("grounding_status", ""),
            "n_source_events": grounding.get("n_source_events", ""),
            "n_source_objects": grounding.get("n_source_objects", ""),
            "n_source_relations": grounding.get("n_source_relations", ""),
        }
        append_row(out_path, result_row)
        done_count += 1
        elapsed = time.time() - t_start
        print(f"[{done_count:02d}/{total}] {qid} ({row.get('query_class','')}/{row.get('difficulty','')}) "
              f"-> {r.status}  den={hit}  {r.latency_s:.1f}s  ETA:{eta_str(elapsed, done_count, total)}")

    print(f"\nSaved: {out_path}")


# ── Evaluation summary ────────────────────────────────────────────────────────

def run_summary() -> None:
    """Refresh the public method comparison summary from saved result CSVs."""
    report_dir = ROOT / "outputs" / "reports"
    method_files = {
        "B1 Zero-shot": {
            "dev": "nb03_b1_results.csv",
            "test": "nb03_b1_results_test.csv",
        },
        "B2 Few-shot": {
            "dev": "nb03_b2_results.csv",
            "test": "nb03_b2_results_test.csv",
        },
        "B3 DIN-SQL": {
            "dev": "nb03_b3_results.csv",
            "test": "nb03_b3_results_test.csv",
        },
        "Method M": {
            "dev": "nb04_method_m_dev.csv",
            "test": "nb04_method_m_test.csv",
        },
    }

    def _load(name: str) -> pd.DataFrame | None:
        path = report_dir / name
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "qid" in df.columns:
            df = df.drop_duplicates(subset="qid", keep="last").reset_index(drop=True)
        return df

    def _exec_rate(df: pd.DataFrame) -> float:
        if "exec_ok" in df.columns:
            return float(df["exec_ok"].astype(float).mean())
        if "status" in df.columns:
            return float((df["status"] == "accept").astype(float).mean())
        return 0.0

    def _join_hall_rate(df: pd.DataFrame) -> float:
        if "join_hall" in df.columns:
            return float(df["join_hall"].fillna(0).astype(float).mean())
        return 0.0

    rows = []
    for method, split_files in method_files.items():
        for split, filename in split_files.items():
            df = _load(filename)
            if df is None or df.empty:
                continue
            rows.append({
                "Method": method,
                "Split": split,
                "n": len(df),
                "ExecRate": _exec_rate(df),
                "DenAcc": float(df["den_acc"].astype(float).mean()),
                "JoinHall": _join_hall_rate(df),
                "AvgLat": float(df["latency_s"].astype(float).mean()),
            })

    if not rows:
        print("Summary skipped: no evaluation result CSVs found.")
        return

    out = report_dir / "gate_e_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Summary saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="all",
                        choices=["all", "b1", "b2", "b3", "method_m"])
    parser.add_argument("--split",   default="dev",
                        choices=["dev", "test"],
                        help="Benchmark split to evaluate (default: dev)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip already-completed questions")
    parser.add_argument("--backend", default="deepseek",
                        choices=["ollama", "deepseek", "openai", "anthropic"],
                        help="LLM backend: deepseek, openai, anthropic, or ollama")
    parser.add_argument("--api_key", default=None,
                        help="API key override (otherwise read from .env)")
    args = parser.parse_args()

    split_file = f"nl2ocel_benchmark_{args.split}.csv"
    df = pd.read_csv(ROOT / "benchmark" / split_file)
    rep = ROOT / "outputs" / "reports"

    suffix = "_test" if args.split == "test" else ""
    bk = "" if args.backend == "deepseek" else f"_{args.backend}"
    out_b1  = rep / f"nb03_b1_results{suffix}{bk}.csv"
    out_b3  = rep / f"nb03_b3_results{suffix}{bk}.csv"
    out_mm  = rep / f"nb04_method_m_{args.split}{bk}.csv"
    out_b2  = rep / f"nb03_b2_results{suffix}{bk}.csv"

    # Mode → output file mapping for B1/B2/B3 evaluation labels:
    #   b1 (zero_shot)  → nb03_b1_results[_test].csv
    #   b2 (few_shot)   → nb03_b2_results[_test].csv   (Nan 2023)
    #   b3 (din_sql)    → nb03_b3_results[_test].csv   (Pourreza 2023)
    kw = {"backend": args.backend, "api_key": args.api_key}
    t0 = time.time()
    if args.mode in ("all", "b1"):
        run_baseline("zero_shot", df, out_b1, args.resume, **kw)
    if args.mode in ("all", "b2"):
        run_baseline("few_shot",  df, out_b2, args.resume, **kw)
    if args.mode in ("all", "method_m"):
        run_method_m(df, out_mm, args.resume, **kw)
    if args.mode in ("all", "b3"):
        run_baseline("din_sql",   df, out_b3, args.resume, **kw)

    if args.mode == "all" and args.split == "dev" and args.backend == "deepseek":
        run_summary()

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
