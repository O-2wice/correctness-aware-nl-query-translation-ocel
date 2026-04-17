"""
nl_to_ir.py  —  Gate D4

NL → Typed IR translator (LLM-backed).

Sends a structured prompt to the LLM (DeepSeek/Ollama) and parses
the response into a validated IR dict.

Includes:
  - system prompt with IR schema definition
  - schema slice injection from SchemaRetriever
  - JSON extraction from LLM response
  - bounded repair loop (re-prompts with VerifyResult errors)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


MAX_REPAIR_ATTEMPTS = 2


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a SQL IR compiler for an Object-Centric SAP O2C event log stored in DuckDB.
The database has three tables:
  - events    (event_id, event_type, timestamp, object_id, object_type)
  - objects   (object_id, object_type, attributes JSON)
  - relations (from_object_id, to_object_id, relation_type)

Your task: translate a natural-language question into a typed IR JSON object.

IR schema (output ONLY valid JSON, no markdown fences, no explanation):
{
  "intent":   "<count_filter|group_topk|temporal_trend|path_relation|delay_analysis|anomaly_filter|conformance|nested_agg|window_agg>",
  "tables":   ["events"|"objects"|"relations"],
  "select":   [{"col": "<col_or_expr>", "agg": "<COUNT|SUM|AVG|MIN|MAX|MEDIAN|STDDEV|QUANTILE_CONT|null>", "alias": "<name>"}],
  "filters":  [{"table": "<table>", "col": "<col>", "op": "<= != > < >= <= IN BETWEEN LIKE>", "val": <value>}],
  "joins":    [{"relation_type": "<whitelist_value>", "from_alias": "<str>", "to_alias": "<str>", "join_col": "<from_object_id|to_object_id>"}],
  "group_by": ["<col_expr>"],
  "order_by": [{"col": "<col>", "dir": "<ASC|DESC>"}],
  "limit":    <int|null>,
  "temporal": {"table": "<table>", "col": "timestamp", "year": <int|null>, "month": <int|null>} | null,
  "ctes":     [{"name": "<cte_name>", "inner": <nested IR>}]
}

RULES:
1. Only use relation_types from the allowed joins list provided.
2. Only reference columns that exist in the schema slice provided.
3. For group_topk add ORDER BY + LIMIT.
4. Use date_diff('day', t1, t2) for day-based delay in DuckDB.
5. Output ONLY the JSON object. No prose, no markdown.

TEMPORAL TREND RULE (intent=temporal_trend):
- Always include the time expression in select AND group_by.
- Use: year(timestamp) for yearly, month(timestamp) for monthly.
- Correct example for "yearly counts":
  "select": [{"col": "year(timestamp)", "agg": null, "alias": "year"},
             {"col": "*", "agg": "COUNT", "alias": "n"}],
  "group_by": ["year(timestamp)"],
  "order_by": [{"col": "year(timestamp)", "dir": "ASC"}]
- NEVER put bare "year" or "month" in group_by without a DuckDB function call.
- For MULTI-YEAR queries ("in 1999, 2000, 2001 and 2002" / "between 1999 and 2002"):
  NEVER use the temporal shorthand field — it only supports a single year.
  Instead use a filter on year(timestamp) with op IN or BETWEEN:
    Multi-year list:  {"col": "year(timestamp)", "op": "IN", "val": [1999, 2000, 2001, 2002]}
    Year range:       {"col": "year(timestamp)", "op": "BETWEEN", "val": [1999, 2002]}
  Combined with GROUP BY year(timestamp) to get a breakdown per year.
  Set temporal: null when using year(timestamp) in filters directly.

GROUP_TOPK RULE (intent=group_topk):
- ALWAYS include the grouping identifier in select AND group_by.
- NEVER output only an aggregation column without the identifier.
- Correct example for "top 5 customers with most dunning events":
  "select": [{"col": "object_id", "agg": null, "alias": "customer_id"},
             {"col": "*", "agg": "COUNT", "alias": "event_count"}],
  "filters": [{"table": "events", "col": "object_type", "op": "=", "val": "customer"}],
  "group_by": ["object_id"],
  "order_by": [{"col": "event_count", "dir": "DESC"}],
  "limit": 5
- For "order items with most X": filter object_type = 'order_item', group by object_id.
- For "customers with most X": filter object_type = 'customer', group by object_id.

DELAY ANALYSIS RULE (intent=delay_analysis):
- Always set joins to the relevant whitelisted relation_type.
- Set select to the requested aggregate over the delay: AVG for average,
  MEDIAN for median, STDDEV for standard deviation, and QUANTILE_CONT for
  percentile questions. For QUANTILE_CONT include "quantile": 0.9 for p90,
  "quantile": 0.99 for p99, etc. in the select item.
- Example for billing→payment clearing delay (invoice date to AR clearing date):
  "joins": [{"relation_type": "billing_to_ar", "from_alias": "bill", "to_alias": "ar", "join_col": "from_object_id"}],
  "select": [{"col": "*", "agg": "AVG", "alias": "avg_delay_days"}]
- IMPORTANT: For billing_to_ar, the compiler uses object attributes FKDAT (billing date on billing_doc)
  and AUGDT (clearing date on ar_item) — NOT event timestamps. This is more accurate for SAP O2C data.
  You do not need to specify the attribute names; just set the join to billing_to_ar.
- The compiler will generate the correct SQL automatically from joins and select.

ANOMALY FILTER RULE (intent=anomaly_filter):
- Set joins to the relevant whitelisted relation_type.
- Set select to COUNT.
- Put the day threshold in filters as: {"col": "delay_days", "op": ">", "val": <number>}
- Example for "more than 873 days":
  "joins": [{"relation_type": "billing_to_ar", "from_alias": "bill", "to_alias": "ar", "join_col": "from_object_id"}],
  "select": [{"col": "*", "agg": "COUNT", "alias": "n_anomalous"}],
  "filters": [{"col": "delay_days", "op": ">", "val": 873}]

EVENT TYPE SYNONYMS (CRITICAL — the schema uses these exact values, not plain English words):
- "invoice" / "invoiced" / "bill" / "billing document" → event_type = "billing_created"
- "payment" / "cleared" / "clearing" / "settled" → event_type = "payment_clearing"
- "order" / "sales order" → event_type = "order_created"
- "delivery" / "shipment" / "shipped" → event_type = "delivery_created"
- "goods issue" / "goods issued" → event_type = "goods_issue"
- "dunning" / "dunning notice" / "overdue notice" → event_type = "dunning_raised"
- "pricing condition" / "price condition" → event_type = "pricing_condition"
Always look up the matching event_type from the schema — never use the English word in the filter value.

SEMANTIC COVERAGE RULE:
- Every explicit business concept in the question must appear in the IR as a filter, object type, or whitelisted join.
- If the question mentions dunning, include event_type = "dunning_raised".
- If it mentions order items linked to customers, include relation_type = "order_to_customer".
- If it mentions order items linked to delivery, include relation_type = "order_to_delivery".
- If it mentions order items linked to billing documents, include relation_type = "order_to_billing".
- If it mentions billing/payment clearing, include relation_type = "billing_to_ar" or event_type = "payment_clearing".
- Do not answer a multi-concept question with a broad count that drops one of those concepts.

PATH NEGATION RULE (intent=path_relation, questions with "never", "no", "without", "not linked"):
When the question asks for objects that are NOT connected to another object type, set top-level "negation": true.
The compiler generates: LEFT JOIN relations ON object_id = from_object_id AND relation_type = X WHERE r.from_object_id IS NULL
Example for "Which order items have no linked delivery?":
  {
    "intent": "path_relation",
    "negation": true,
    "tables": ["objects"],
    "select": [{"col": "object_id", "agg": "COUNT", "alias": "n", "distinct": true}],
    "filters": [{"table": "objects", "col": "object_type", "op": "=", "val": "order_item"}],
    "joins": [{"relation_type": "order_to_delivery", "from_alias": "ord", "to_alias": "del", "join_col": "from_object_id"}],
    "group_by": [], "order_by": [], "limit": null, "temporal": null, "ctes": []
  }

DISTINCT COUNT RULE:
When counting distinct objects (not raw event rows), add "distinct": true to the select item.
Example: "How many distinct customers placed orders?"
  "select": [{"col": "object_id", "agg": "COUNT", "alias": "n_customers", "distinct": true}]
Use distinct=true whenever the question says "how many [object type]" and could have multiple events per object.

CONFORMANCE RULE (intent=conformance) — "never", "without", "orphan", "missing":
Used when the question asks how many objects LACK a specific event, LACK a relation, or have
NULL attributes. The IR adds a `check_type` field plus check-specific fields:

  check_type = "missing_event"   — e.g., "order items that never had a goods_issue event"
      {"intent": "conformance", "tables": ["objects"], "select": [],
       "check_type": "missing_event",
       "base": {"object_type": "order_item"},
       "missing_event_type": "goods_issue"}

  check_type = "missing_relation" — e.g., "delivery items not linked to a billing document"
      {"intent": "conformance", "tables": ["objects"], "select": [],
       "check_type": "missing_relation",
       "base": {"object_type": "delivery_item"},
       "missing_relation_type": "delivery_to_billing"}

  check_type = "set_difference_relations" — e.g., "order items billed but not delivered"
      {"intent": "conformance", "tables": ["relations"], "select": [],
       "check_type": "set_difference_relations",
       "base": {"object_type": "order_item"},
       "has_relation_type": "order_to_billing",
       "also_missing_relation": "order_to_delivery"}

  check_type = "attribute_null"  — e.g., "billing documents with missing ZTERM"
      {"intent": "conformance", "tables": ["objects"], "select": [],
       "check_type": "attribute_null",
       "base": {"object_type": "billing_doc"},
       "attribute_null_column": "ZTERM"}
      For multi-column NULL conditions, add extra_null_columns list and attribute_null_logic
      ("or" | "and"). Example: NETWR or WAERK missing:
        "extra_null_columns": ["WAERK"], "attribute_null_logic": "or"

NESTED_AGG RULE (intent=nested_agg) — correlated sub-queries with population statistics:
Used when the question compares a grouped aggregate to a summary stat of the SAME aggregate
(e.g., "which event types have above-average counts?"). The IR declares the group/metric and
a comparison against AVG / MEDIAN / MAX / MIN of that metric:

  Example: "Which event types had above-average counts in 2005?"
      {"intent": "nested_agg", "tables": ["events"], "select": [],
       "base": {"table": "events", "group_by": "event_type"},
       "metric": {"agg": "COUNT", "col": "*"},
       "comparison": {"op": ">", "reference": "AVG"},
       "filters": [{"table": "events", "col": "year(timestamp)", "op": "=", "val": 2005}]}

  For the "how many X pass the filter?" variant, add an outer select aggregation:
       "select": [{"agg": "COUNT", "alias": "n"}]

  For reference to a SPECIFIC row rather than population statistic (e.g., "more than 2004"):
       "comparison": {"op": ">", "reference": "REFERENCE_VALUE"},
       "reference_value": 2004, "reference_group_key": "year"

WINDOW_AGG RULE (intent=window_agg) — rolling averages, cumulative sums, ranks, LAG / LEAD:
Used when the question asks for a secondary per-row computation over an ordered sequence.
The IR carries a `base` aggregate plus a `window` spec:

  Example: "3-month trailing moving average of billing_created counts in 2005"
      {"intent": "window_agg", "tables": ["events"], "select": [],
       "base": {"table": "events", "order_by": "month"},
       "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
       "window": {"function": "AVG", "order_by": "month",
                  "rows_preceding": 2, "alias": "moving_avg"},
       "filters": [{"table": "events", "col": "event_type", "op": "=", "val": "billing_created"},
                   {"table": "events", "col": "year(timestamp)", "op": "=", "val": 2005}]}

  Supported window.function values:
    AVG / SUM  — with rows_preceding for rolling / cumulative (omit rows_preceding for running total)
    LAG / LEAD — with integer `offset` for period-over-period delta
    RANK / DENSE_RANK / ROW_NUMBER / PERCENT_RANK — for ordering without aggregation on metric

  Use `window.partition_by` when the window should reset per group (e.g., per object_type rank).
"""

_REPAIR_PROMPT_TEMPLATE = """\
Your previous IR had errors. Fix ONLY the issues listed below and output corrected IR JSON.

Errors:
{errors}

Repair hints:
{hints}

Previous IR:
{prev_ir}

Question: {question}
{schema_slice}

Output ONLY corrected JSON.
"""


# ── LLM caller — delegates to shared llm_client ───────────────────────────────

def _call_llm(prompt: str, backend: str, model: str,
              api_key: str | None = None) -> str:
    from .llm_client import call_llm
    result = call_llm(prompt, backend=backend, model=model, api_key=api_key)
    if result["error"]:
        raise RuntimeError(result["error"])
    return result["text"]


# ── JSON extractor ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extract first JSON object from LLM response.
    Handles markdown fences and leading prose.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find first { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("Unmatched braces in LLM response")


# ── Main translator ────────────────────────────────────────────────────────────

class NLtoIRTranslator:
    """
    NL → typed IR translator with repair loop.

    Usage:
        translator = NLtoIRTranslator(retriever, verifier_fn, backend='deepseek')
        result = translator.translate(question)
    """

    def __init__(
        self,
        retriever,          # SchemaRetriever instance
        verify_fn,          # callable(ir) → VerifyResult
        backend: str = "deepseek",
        model:   str = "deepseek-chat",
        api_key: str | None = None,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    ):
        self._retriever  = retriever
        self._verify     = verify_fn
        self._backend    = backend
        self._model      = model
        self._api_key    = api_key
        self._max_repair_attempts = max(0, int(max_repair_attempts))

    def translate(self, question: str, k: int = 12) -> dict:
        """
        Translate NL question to verified IR dict.

        Returns:
            {
              "ir":           dict | None,
              "status":       "accept" | "reject" | "llm_error" | "parse_error",
              "attempts":     int,
              "errors":       list[str],
              "latency_s":    float,
              "schema_slice": dict,
            }
        """
        t0 = time.perf_counter()
        schema_slice = self._retriever.retrieve(question, k=k)
        schema_text  = self._retriever.to_prompt_text(question, k=k)

        full_prompt = (
            _SYSTEM_PROMPT
            + f"\n\n{schema_text}\n\n"
            + f"Question: {question}\n"
        )

        ir            = None
        verify_result = None
        errors:   list[str] = []
        attempts  = 0
        status    = "llm_error"

        for attempt in range(1 + self._max_repair_attempts):
            attempts = attempt + 1
            try:
                if attempt == 0:
                    raw = _call_llm(full_prompt, self._backend, self._model, self._api_key)
                else:
                    repair_hint = verify_result.repair_hint if verify_result is not None else {}
                    repair_prompt = _REPAIR_PROMPT_TEMPLATE.format(
                        errors="\n".join(errors),
                        hints=json.dumps(repair_hint, indent=2),
                        prev_ir=json.dumps(ir, indent=2),
                        question=question,
                        schema_slice=schema_text,
                    )
                    raw = _call_llm(repair_prompt, self._backend, self._model, self._api_key)
            except Exception as exc:
                status = "llm_error"
                errors = [str(exc)]
                break

            try:
                ir = _extract_json(raw)
            except Exception as exc:
                status = "parse_error"
                errors = [f"JSON parse failed: {exc}", f"Raw: {raw[:300]}"]
                continue

            verify_result = self._verify(ir)
            if verify_result.ok:
                status = "accept"
                errors = []
                break

            status = verify_result.status
            errors = verify_result.errors
            if verify_result.status == "reject":
                break  # structural — repair won't help

        # Repair budget exhausted without acceptance → treat as reject
        if status not in ("accept", "reject", "llm_error", "parse_error"):
            status = "reject"

        latency = time.perf_counter() - t0
        return {
            "ir":           ir if status == "accept" else None,
            "status":       status,
            "attempts":     attempts,
            "errors":       errors,
            "latency_s":    round(latency, 3),
            "schema_slice": schema_slice,
        }


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from nl2ocel.llm_client import api_key_for_backend, default_model_for_backend

    root = Path(__file__).resolve().parents[2]

    from nl2ocel.schema_retriever import SchemaRetriever
    from nl2ocel.query_verifier   import verify_ir, load_schema_index, load_whitelist_set

    retriever     = SchemaRetriever.from_configs(
        root / "configs" / "schema_catalog.json",
        root / "configs" / "relation_whitelist.json",
    )
    schema_index  = load_schema_index(root / "configs" / "schema_catalog.json")
    allowed_joins = load_whitelist_set(root / "configs" / "relation_whitelist.json")

    def _verify(ir):
        return verify_ir(ir, schema_index, allowed_joins)

    backend = os.environ.get("NL2OCEL_BACKEND") or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "ollama")
    backend = backend.strip().lower()
    model = os.environ.get("NL2OCEL_MODEL") or default_model_for_backend(backend)
    credential = api_key_for_backend(backend, os.environ.get("NL2OCEL_API_KEY"))
    translator = NLtoIRTranslator(retriever, _verify, backend=backend, model=model, api_key=credential)

    q = "How many billing documents were created in 2019?"
    print(f"Q: {q}")
    result = translator.translate(q)
    print(f"Status: {result['status']}  attempts: {result['attempts']}  latency: {result['latency_s']}s")
    if result["ir"]:
        print(json.dumps(result["ir"], indent=2))
    else:
        print(f"Errors: {result['errors']}")
