"""
query_verifier.py  —  Gate D2

IR Verifier: V(z, s, S, P) → {accept, reject, repair}

Checks a typed IR dict against:
  1. Schema validity  — all table/col refs exist in schema_catalog
  2. Join whitelist   — all relation_types in whitelist
  3. Safety policy    — no write ops, no blocked event/object types
  4. Structural rules — required keys present, intent is known

On failure: returns VerifyResult with status='reject' and a repair_hint dict
that the caller can pass back to the LLM for a correction attempt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


KNOWN_INTENTS = {
    # Phase 1 — six original intents.
    "count_filter",
    "group_topk",
    "temporal_trend",
    "path_relation",
    "delay_analysis",
    "anomaly_filter",
    # Phase 2 — IR extensions for OCEL analytical patterns surfaced by the
    # 120-question benchmark. See src/nl2ocel/ir_to_sql.py for the per-intent
    # IR shapes and the specialised compilers that handle them.
    "conformance",    # NOT IN / NOT EXISTS / attribute-null checks
    "nested_agg",     # correlated sub-query filters (x > AVG(x))
    "window_agg",     # OVER / PARTITION BY / ROWS BETWEEN / LAG / RANK
}

KNOWN_TABLES = {"events", "objects", "relations"}

KNOWN_OPS = {"=", "!=", ">", "<", ">=", "<=", "IN", "BETWEEN", "LIKE", "IS NULL"}

# Virtual columns computed inside multi-CTE templates — not in raw schema but
# valid. Phase-2 intents (nested_agg, window_agg) emit metric / window aliases
# like `n`, `moving_avg`, `cumulative`, `yoy_change` that the verifier must
# also accept.
VIRTUAL_COLS = {
    # Phase-1 delay/anomaly cols
    "delay_days", "n_days", "diff_days", "day_diff",
    # Phase-2 common metric / window aliases
    "n", "total", "moving_avg", "cumulative", "yoy_change", "proportion",
    "rank", "n_events", "span_days", "avg_span_days", "avg_delay",
    "avg_days", "median_days", "max_days", "min_days", "p90_days", "p99_days",
    "stddev_days", "running_total", "cleared_share", "cleared_rate",
}

# Intents that build their own CTE-and-alias structure, bypassing the
# standard SELECT/FROM/WHERE template. The verifier therefore should not
# try to map their select.col entries back to raw schema columns, because
# by construction those are CTE output aliases.
_ALIAS_ONLY_INTENTS = {"nested_agg", "window_agg", "conformance"}

KNOWN_AGGS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "MEDIAN", "STDDEV",
    "QUANTILE_CONT", "PERCENTILE_CONT", None,
}


@dataclass
class VerifyResult:
    status: str            # 'accept' | 'reject' | 'repair'
    errors: list[str] = field(default_factory=list)
    repair_hint: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "accept"


def load_schema_index(catalog_path: str | Path) -> dict:
    """
    Build col-lookup from schema_catalog.json.
    Returns {table_name: set(col_names)}.
    """
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    index: dict[str, set] = {}
    for tbl in catalog.get("tables", []):
        index[tbl["table_name"]] = {c["name"] for c in tbl.get("columns", [])}
    return index


def load_enum_values(catalog_path: str | Path) -> dict[str, set]:
    """
    Build {col_name: set(valid_values)} for enumerable columns.
    Used to validate filter values for event_type, object_type, relation_type.
    """
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    enums: dict[str, set] = {}
    for tbl in catalog.get("tables", []):
        for col in tbl.get("columns", []):
            vals = col.get("sample_values")
            if vals and col["name"] in ("event_type", "object_type", "relation_type"):
                enums.setdefault(col["name"], set()).update(vals)
    return enums


def load_whitelist_set(whitelist_path: str | Path) -> set:
    data = json.loads(Path(whitelist_path).read_text(encoding="utf-8"))
    return {w["relation_type"] for w in data["whitelist"]}


def load_guardrail_policy(policy_path: str | Path) -> dict:
    """Load guardrail_policy.yaml — returns dict with blocklists and flags."""
    try:
        import yaml  # type: ignore
        with open(policy_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ── Main verifier ─────────────────────────────────────────────────────────────

def verify_ir(
    ir: dict,
    schema_index: dict[str, set],
    allowed_joins: set[str],
    policy: dict | None = None,
    enum_values: dict[str, set] | None = None,
) -> VerifyResult:
    """
    V(z, s, S, P) → VerifyResult

    Args:
        ir:           typed IR dict
        schema_index: {table: set(cols)} from schema_catalog
        allowed_joins: set of valid relation_types from whitelist
        policy:       optional guardrail policy dict

    Returns VerifyResult with status in {accept, reject, repair}.
    'repair' means errors are fixable by substitution;
    'reject' means the IR is structurally broken.
    """
    errors: list[str] = []
    repair_hints: dict[str, Any] = {}

    # ── 1. Required keys ──────────────────────────────────────────────────────
    for key in ("intent", "tables", "select"):
        if key not in ir:
            errors.append(f"Missing required IR key: '{key}'")

    if errors:
        return VerifyResult(status="reject", errors=errors)

    # ── 2. Intent ─────────────────────────────────────────────────────────────
    intent = ir.get("intent", "")
    if intent not in KNOWN_INTENTS:
        errors.append(f"Unknown intent '{intent}'. Must be one of {sorted(KNOWN_INTENTS)}")
        repair_hints["intent"] = f"Use one of: {sorted(KNOWN_INTENTS)}"

    policy_allowed_tables = set(policy.get("allowed_tables", [])) if policy else set()
    policy_allowed_aggs = set(policy.get("allowed_aggregations", [])) if policy else set()
    policy_allowed_ops = set(policy.get("allowed_predicates", [])) if policy else set()

    # ── 3. Tables ─────────────────────────────────────────────────────────────
    for tbl in ir.get("tables", []):
        if tbl not in KNOWN_TABLES:
            errors.append(f"Unknown table '{tbl}'. Must be one of {sorted(KNOWN_TABLES)}")
            repair_hints.setdefault("tables", f"Only {sorted(KNOWN_TABLES)} allowed")
        if policy_allowed_tables and tbl not in policy_allowed_tables:
            errors.append(f"Table '{tbl}' is not allowed by guardrail policy")
            repair_hints.setdefault("tables", f"Only {sorted(policy_allowed_tables)} allowed")

    # ── 4. Select columns ─────────────────────────────────────────────────────
    # Phase-2 alias-only intents build their own CTE aliases — skip deep
    # per-column schema checks for them. The verifier still checks the agg
    # keyword below.
    skip_col_check = intent in _ALIAS_ONLY_INTENTS
    for s in ir.get("select", []):
        col = s.get("col", "")
        agg = s.get("agg")
        if agg is not None and agg.upper() not in {a for a in KNOWN_AGGS if a}:
            errors.append(f"Unknown aggregation '{agg}'")
            repair_hints.setdefault("select_agg", f"Use one of {[a for a in KNOWN_AGGS if a]}")
        if agg is not None and policy_allowed_aggs and agg.upper() not in policy_allowed_aggs:
            errors.append(f"Aggregation '{agg}' is not allowed by guardrail policy")
            repair_hints.setdefault("select_agg", f"Use one of {sorted(policy_allowed_aggs)}")
        # skip deep col validation for expressions like year(timestamp)
        if skip_col_check:
            continue
        if col and "(" not in col and col != "*":
            found = any(col.split(".")[-1] in cols for cols in schema_index.values())
            if not found:
                errors.append(f"Column '{col}' not found in any table schema")
                repair_hints.setdefault("hallucinated_cols", []).append(col)

    # ── 5. Filter columns ─────────────────────────────────────────────────────
    for f in ir.get("filters", []):
        col = f.get("col", "")
        op  = f.get("op", "").upper()
        val = f.get("val")
        if op not in KNOWN_OPS:
            errors.append(f"Unknown filter op '{op}'")
        if policy_allowed_ops and op not in policy_allowed_ops:
            errors.append(f"Filter op '{op}' is not allowed by guardrail policy")
            repair_hints.setdefault("filter_ops", f"Use one of {sorted(policy_allowed_ops)}")
        if col and "(" not in col:
            bare_col = col.split(".")[-1]
            if bare_col not in VIRTUAL_COLS:
                found = any(bare_col in cols for cols in schema_index.values())
                if not found:
                    errors.append(f"Filter column '{col}' not in schema")
                    repair_hints.setdefault("hallucinated_cols", []).append(col)
                elif enum_values and bare_col in enum_values and isinstance(val, str):
                    if val not in enum_values[bare_col]:
                        valid = sorted(enum_values[bare_col])
                        errors.append(
                            f"Filter value '{val}' is not a valid {bare_col}. "
                            f"Valid values: {valid}"
                        )
                        repair_hints.setdefault("invalid_enum_values", {})[bare_col] = valid

    # ── 6. Join whitelist ─────────────────────────────────────────────────────
    bad_joins: list[str] = []
    for j in ir.get("joins", []):
        rt = j.get("relation_type", "")
        if rt not in allowed_joins:
            bad_joins.append(rt)
            errors.append(
                f"Join '{rt}' not in whitelist. Allowed: {sorted(allowed_joins)}"
            )
    if bad_joins:
        repair_hints["bad_joins"] = bad_joins
        repair_hints["allowed_joins"] = sorted(allowed_joins)

    # ── 7. Safety policy ──────────────────────────────────────────────────────
    if policy:
        blocked_evt = set(policy.get("blocked_event_types", []))
        blocked_obj = set(policy.get("blocked_object_types", []))

        for f in ir.get("filters", []):
            val = f.get("val")
            if isinstance(val, str) and val in blocked_evt:
                errors.append(f"Filter references blocked event_type '{val}'")
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v in blocked_evt:
                        errors.append(f"Filter references blocked event_type '{v}'")

    # ── 7b. anomaly_filter must have a numeric threshold AND a join ───────────
    if intent == "anomaly_filter":
        has_threshold = any(
            isinstance(f.get("val"), (int, float)) and f.get("val", 0) > 0
            for f in ir.get("filters", [])
        )
        if not has_threshold:
            errors.append(
                "anomaly_filter IR must include a numeric threshold filter "
                "e.g. {\"col\": \"delay_days\", \"op\": \">\", \"val\": <number>}"
            )
            repair_hints["anomaly_filter_threshold"] = (
                "Add a filter: {\"col\": \"delay_days\", \"op\": \">\", \"val\": <the threshold number from the question>}"
            )
        if not ir.get("joins"):
            errors.append(
                "anomaly_filter IR must include a join with a whitelisted relation_type. "
                f"Allowed: {sorted(allowed_joins)}"
            )
            repair_hints["anomaly_filter_join"] = (
                f"Add a join: {{\"relation_type\": \"<one of {sorted(allowed_joins)}>\", "
                "\"from_alias\": \"e1\", \"to_alias\": \"e2\", \"join_col\": \"from_object_id\"}}"
            )

    # ── 8. CTE recursion ──────────────────────────────────────────────────────
    for cte in ir.get("ctes", []):
        inner = cte.get("inner", {})
        if inner:
            inner_result = verify_ir(inner, schema_index, allowed_joins, policy)
            if not inner_result.ok:
                errors.extend([f"CTE '{cte['name']}': {e}" for e in inner_result.errors])
                repair_hints.update(inner_result.repair_hint)

    # ── Decision ──────────────────────────────────────────────────────────────
    if not errors:
        return VerifyResult(status="accept")

    # Classify: repair if only join/col hallucinations, reject if structural
    structural = [e for e in errors if "Missing required" in e or "Unknown intent" in e]
    status = "reject" if structural else "repair"

    return VerifyResult(status=status, errors=errors, repair_hint=repair_hints)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    schema_index = load_schema_index(root / "configs" / "schema_catalog.json")
    allowed_joins = load_whitelist_set(root / "configs" / "relation_whitelist.json")

    good_ir = {
        "intent": "count_filter",
        "tables": ["events"],
        "select": [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters": [{"table": "events", "col": "event_type", "op": "=", "val": "billing_created"}],
        "joins": [], "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }

    bad_ir = {
        "intent": "path_relation",
        "tables": ["events", "ghost_table"],
        "select": [{"col": "ghost_col", "agg": None, "alias": "x"}],
        "filters": [],
        "joins": [{"relation_type": "fake_join", "from_alias": "a", "to_alias": "b", "join_col": "from_object_id"}],
        "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }

    for label, ir in [("GOOD IR", good_ir), ("BAD IR", bad_ir)]:
        result = verify_ir(ir, schema_index, allowed_joins)
        print(f"\n{label}: {result.status}")
        for e in result.errors:
            print(f"  ERROR: {e}")
        if result.repair_hint:
            print(f"  REPAIR HINTS: {result.repair_hint}")
