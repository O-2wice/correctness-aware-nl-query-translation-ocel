"""
benchmark_dedup_analysis.py

Reports template accuracy vs surface accuracy for the benchmark.

A "template" is a unique gold SQL pattern. Multiple NL questions may share
the same gold SQL (paraphrastic duplicates). This inflates the benchmark
effective size and can bias DenAcc estimates.

Output:
  - Template count vs surface count per split
  - List of duplicate template groups
  - Template DenAcc vs surface DenAcc per method (dev set)

Usage:
    python scripts/benchmark_dedup_analysis.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
BM      = ROOT / "benchmark"
REPORTS = ROOT / "outputs" / "reports"


def sql_canonical(sql: str) -> str:
    """Normalise SQL to a canonical form for dedup: strip whitespace, lowercase."""
    return " ".join(sql.lower().split())


def sql_hash(sql: str) -> str:
    return hashlib.sha256(sql_canonical(sql).encode()).hexdigest()[:12]


def load_split(split: str) -> pd.DataFrame:
    path = BM / f"nl2ocel_benchmark_{split}.csv"
    df = pd.read_csv(path)
    df["sql_hash"] = df["gold_sql"].apply(sql_hash)
    return df


def dedup_report(df: pd.DataFrame, split: str) -> dict:
    total_q   = len(df)
    templates = df["sql_hash"].nunique()
    dup_ratio = 1 - templates / total_q

    groups = df.groupby("sql_hash").agg(
        n_questions=("qid", "count"),
        qids=("qid", lambda x: ", ".join(sorted(x))),
        gold_sql=("gold_sql", "first"),
        query_class=("query_class", "first"),
    ).reset_index()

    duplicated = groups[groups["n_questions"] > 1].sort_values("n_questions", ascending=False)

    return {
        "split":      split,
        "n_questions": total_q,
        "n_templates": templates,
        "dup_ratio":   dup_ratio,
        "duplicated":  duplicated,
    }


def template_den_acc(bm_df: pd.DataFrame, result_path: Path, method: str) -> dict | None:
    if not result_path.exists():
        return None
    res = pd.read_csv(result_path)
    if "qid" in res.columns:
        res = res.drop_duplicates(subset="qid", keep="last")

    merged = bm_df.merge(res[["qid", "den_acc"]], on="qid", how="left")
    merged["den_acc"] = merged["den_acc"].fillna(0)

    surface_acc  = merged["den_acc"].mean()
    template_acc = merged.groupby("sql_hash")["den_acc"].max().mean()

    return {
        "method":       method,
        "surface_acc":  surface_acc,
        "template_acc": template_acc,
        "gap":          template_acc - surface_acc,
    }


def main():
    for split in ("dev", "test"):
        bm = load_split(split)
        r  = dedup_report(bm, split)

        print(f"\n{'='*60}")
        print(f"Split: {split.upper()}  ({r['n_questions']} questions, "
              f"{r['n_templates']} unique templates, "
              f"{r['dup_ratio']:.0%} duplication rate)")
        print(f"{'='*60}")

        if len(r["duplicated"]) > 0:
            print("\nDuplicate template groups:")
            for _, row in r["duplicated"].iterrows():
                print(f"  [{row['query_class']}] {row['qids']}  "
                      f"({row['n_questions']} questions, same SQL)")
                print(f"    SQL: {row['gold_sql'][:80]}...")
        else:
            print("  No duplicate templates.")

    # ── Template vs surface DenAcc for Method M (dev) ─────────────────────────
    print(f"\n{'='*60}")
    print("Template DenAcc vs Surface DenAcc (dev, Method M)")
    print(f"{'='*60}")
    bm_dev = load_split("dev")
    bm_dev["sql_hash"] = bm_dev["gold_sql"].apply(sql_hash)

    method_files = {
        "Method M":   REPORTS / "nb04_method_m_dev.csv",
        "B2 Few-shot": REPORTS / "nb03_b2_results.csv",
        "B3 DIN-SQL":  REPORTS / "nb03_b3_results.csv",
    }
    rows = []
    for method, path in method_files.items():
        r = template_den_acc(bm_dev, path, method)
        if r:
            rows.append(r)
            print(f"  {method:15s}  surface={r['surface_acc']:.1%}  "
                  f"template={r['template_acc']:.1%}  gap={r['gap']:+.1%}")

    if rows:
        out = REPORTS / "benchmark_dedup_analysis.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
