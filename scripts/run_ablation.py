"""
run_ablation.py

Ablation study: isolates the contribution of each pipeline component.

Variants tested against the dev set:
  M-full     — full Constrained pipeline (retriever + verifier + repair + compiler)
  M-noretriever — full schema injected instead of top-k slice (k=∞)
  M-norepair — repair loop disabled (max 1 LLM attempt)
  M-noverifier — verifier skipped (IR accepted on first parse success)

Metrics: ExecRate, DenAcc, JoinHall, AvgLatency, AvgAttempts

Usage:
    python scripts/run_ablation.py --backend deepseek --split dev --api_key <key>
    python scripts/run_ablation.py --backend ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2ocel.llm_client import api_key_for_backend, default_model_for_backend
from nl2ocel.schema_retriever import SchemaRetriever
from nl2ocel.query_verifier import (
    verify_ir, load_schema_index, load_whitelist_set,
    load_sql_policy, load_enum_values, VerifyResult,
)
from nl2ocel.nl_to_ir import NLtoIRTranslator
from nl2ocel.pipeline import ConstrainedPipeline

WHITELIST = {
    "billing_to_ar", "order_to_delivery", "order_to_billing",
    "delivery_to_billing", "order_to_customer", "order_to_material", "order_to_payer",
}


def _build_pipeline(
    backend: str,
    model: str,
    api_key: str | None,
    *,
    full_schema: bool = False,
    no_repair: bool = False,
    no_verifier: bool = False,
) -> ConstrainedPipeline:
    retriever     = SchemaRetriever.from_configs(
        ROOT / "configs" / "schema_catalog.json",
        ROOT / "configs" / "relation_whitelist.json",
    )
    schema_index  = load_schema_index(ROOT / "configs" / "schema_catalog.json")
    allowed_joins = load_whitelist_set(ROOT / "configs" / "relation_whitelist.json")
    policy        = load_sql_policy(ROOT / "configs" / "sql_policy.yaml")
    enum_values   = load_enum_values(ROOT / "configs" / "schema_catalog.json")

    if no_verifier:
        def _verify(ir): return VerifyResult(status="accept")
    else:
        def _verify(ir): return verify_ir(ir, schema_index, allowed_joins, policy, enum_values)

    # Monkey-patch retriever to return full schema slice if full_schema=True
    if full_schema:
        _orig_retrieve = retriever.retrieve
        retriever.retrieve = lambda q, k=12: _orig_retrieve(q, k=9999)
        _orig_text = retriever.to_prompt_text
        retriever.to_prompt_text = lambda q, k=12: _orig_text(q, k=9999)

    translator = NLtoIRTranslator(
        retriever=retriever,
        verify_fn=_verify,
        backend=backend,
        model=model,
        api_key=api_key,
        max_repair_attempts=0 if no_repair else 2,
    )

    return ConstrainedPipeline(
        retriever=retriever,
        translator=translator,
        schema_index=schema_index,
        allowed_joins=allowed_joins,
        policy=policy,
        ocel_path=ROOT / "data" / "processed" / "ocel",
    )


def run_variant(
    variant_name: str,
    pipeline: ConstrainedPipeline,
    bm: pd.DataFrame,
    allowed_joins: set,
) -> list[dict]:
    rows = []
    n = len(bm)
    for i, (_, row) in enumerate(bm.iterrows(), 1):
        qid      = row["qid"]
        question = row["nl_question"]
        gold_hash = row["gold_result_hash"]

        result = pipeline.run(question)
        status = "reject" if result.status == "repair" else result.status

        joins_used = result.provenance.get("joins_used", [])
        join_hall  = int(any(j not in allowed_joins for j in joins_used))

        den_acc = int(result.result_hash == gold_hash) if result.result_hash else 0
        print(f"  [{i:02d}/{n}] {qid} -> {status}  den={den_acc}  {result.latency_s:.1f}s")

        rows.append({
            "variant":     variant_name,
            "qid":         qid,
            "query_class": row["query_class"],
            "difficulty":  row["difficulty"],
            "status":      status,
            "den_acc":     den_acc,
            "exec_ok":     int(status == "accept"),
            "join_hall":   join_hall,
            "ir_attempts": result.ir_attempts,
            "latency_s":   result.latency_s,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {
        "variant":     df["variant"].iloc[0],
        "n":           len(df),
        "ExecRate":    df["exec_ok"].mean(),
        "DenAcc":      df["den_acc"].mean(),
        "JoinHall":    df["join_hall"].mean(),
        "AvgLatency":  df["latency_s"].mean(),
        "AvgAttempts": df["ir_attempts"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["deepseek", "openai", "anthropic", "ollama"], default="deepseek")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--model",   default="")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Benchmark split to evaluate (default: dev)")
    parser.add_argument("--variants", nargs="+",
                        default=["M-full", "M-noretriever", "M-norepair", "M-noverifier"],
                        help="Which ablation variants to run")
    args = parser.parse_args()

    model = args.model or default_model_for_backend(args.backend)
    credential = api_key_for_backend(args.backend, args.api_key or None)

    bm = pd.read_csv(ROOT / "benchmark" / f"nl2ocel_benchmark_{args.split}.csv")
    allowed_joins = load_whitelist_set(ROOT / "configs" / "relation_whitelist.json")

    VARIANTS = {
        "M-full":        dict(full_schema=False, no_repair=False, no_verifier=False),
        "M-noretriever": dict(full_schema=True,  no_repair=False, no_verifier=False),
        "M-norepair":    dict(full_schema=False, no_repair=True,  no_verifier=False),
        "M-noverifier":  dict(full_schema=False, no_repair=False, no_verifier=True),
    }

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for variant in args.variants:
        if variant not in VARIANTS:
            print(f"Unknown variant '{variant}', skipping")
            continue
        print(f"\n{'='*55}\nVariant: {variant}\n{'='*55}")
        pipeline = _build_pipeline(
            args.backend, model, credential, **VARIANTS[variant]
        )
        try:
            rows = run_variant(variant, pipeline, bm, allowed_joins)
        finally:
            pipeline.close()
        for row in rows:
            row["split"] = args.split
        all_rows.extend(rows)
        s = summarise(rows)
        summaries.append(s)
        print(f"  -> ExecRate={s['ExecRate']:.1%}  DenAcc={s['DenAcc']:.1%}  "
              f"JoinHall={s['JoinHall']:.1%}  Latency={s['AvgLatency']:.1f}s  "
              f"Attempts={s['AvgAttempts']:.2f}")

    print(f"\n{'='*55}\nABLATION SUMMARY\n{'='*55}")
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    suffix = "" if args.split == "dev" else f"_{args.split}"
    out_detail  = ROOT / "outputs" / "reports" / f"ablation_detail{suffix}.csv"
    out_summary = ROOT / "outputs" / "reports" / f"ablation_summary{suffix}.csv"
    pd.DataFrame(all_rows).to_csv(out_detail, index=False)
    summary_df.to_csv(out_summary, index=False)
    print(f"\nSaved: {out_detail}")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
