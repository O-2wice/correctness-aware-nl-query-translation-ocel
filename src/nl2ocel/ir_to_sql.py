"""
ir_to_sql.py  —  Gate D1

Typed IR → DuckDB SQL compiler.

The IR is a constrained intermediate representation. This module compiles it
to SQL without touching the LLM. All join whitelist enforcement happens here —
any IR that requests a join not in the whitelist raises IRCompileError.

IR schema (dict):
{
  "intent":   str,          # count_filter | group_topk | temporal_trend |
                            # path_relation | delay_analysis | anomaly_filter |
                            # conformance | nested_agg | window_agg
  "tables":   list[str],    # subset of {events, objects, relations}
  "select":   list[{        # columns/aggregations to return
                "col":   str,
                "agg":   str | None,   # COUNT SUM AVG MIN MAX or None
                "alias": str
             }],
  "filters":  list[{        # WHERE conditions
                "table": str,
                "col":   str,
                "op":    str,          # = != > < >= <= IN BETWEEN LIKE
                "val":   any
             }],
  "joins":    list[{        # object-centric joins via relations table
                "relation_type":  str,   # MUST be in whitelist
                "from_alias":     str,   # alias for from_object_type events/objects
                "to_alias":       str,   # alias for to_object_type events/objects
                "join_col":       str    # from_object_id or to_object_id
             }],
  "group_by": list[str],    # column expressions
  "order_by": list[{        # sort specification
                "col": str,
                "dir": str  # ASC | DESC
             }],
  "limit":    int | None,
  "temporal": {             # optional temporal filter shorthand
                "table": str,
                "col":   str,
                "year":  int | None,
                "month": int | None
             } | None,
  "ctes":     list[{        # WITH clauses for NESTED queries
                "name":  str,
                "inner": dict   # nested IR dict
             }]
}
"""

import json
import re
from pathlib import Path
from typing import Any


class IRCompileError(Exception):
    pass


# ── Temporal expression normalisation map ─────────────────────────────────────
# LLMs often emit bare alias names ("year", "month") instead of DuckDB
# expressions. This map converts them so GROUP BY / ORDER BY are valid.
_TEMPORAL_ALIAS_MAP = {
    "year":       "year(timestamp)",
    "month":      "month(timestamp)",
    "quarter":    "quarter(timestamp)",
    "year_month": "strftime(timestamp, '%Y-%m')",
    "week":       "week(timestamp)",
    "day":        "day(timestamp)",
}

# Default event-type → object-type for OCEL delay/anomaly queries.
# Used when the IR does not explicitly carry object_type filters.
_DEFAULT_OBJ_TYPE = {
    "billing_created":  "billing_doc",
    "payment_clearing": "ar_item",
    "order_created":    "order_item",
    "delivery_created": "delivery_item",
    "goods_issue":      "delivery_item",
    "create_billing_document": "billing_doc",
    "clear_invoice":    "ar_item",
}

SUPPORTED_AGGS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "MEDIAN", "STDDEV",
    "QUANTILE_CONT", "PERCENTILE_CONT",
}


def load_whitelist(path: str | Path) -> set:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {w["relation_type"] for w in data["whitelist"]}


def _infer_quantile(select_item: dict) -> float:
    """Read a percentile value from IR, or infer p90/p99 from the alias."""
    for key in ("quantile", "q", "percentile"):
        val = select_item.get(key)
        if isinstance(val, (int, float)):
            return float(val) / 100.0 if float(val) > 1 else float(val)
    alias_text = " ".join(
        str(select_item.get(k, "")) for k in ("alias", "col", "name")
    ).lower()
    match = re.search(r"\bp(?:ercentile_?)?(\d{1,2})\b", alias_text)
    if match:
        return int(match.group(1)) / 100.0
    raise IRCompileError(
        "QUANTILE_CONT/PERCENTILE_CONT requires quantile, q, percentile, "
        "or an alias like p90_days"
    )


def _agg_expr(agg: str, inner: str, select_item: dict | None = None) -> str:
    """Render one aggregate expression, including percentile aggregates."""
    agg_norm = (agg or "").upper()
    item = select_item or {}
    if agg_norm == "PERCENTILE_CONT":
        agg_norm = "QUANTILE_CONT"
    if agg_norm == "QUANTILE_CONT":
        q = _infer_quantile(item)
        return f"QUANTILE_CONT({inner}, {q:g})"
    if agg_norm not in SUPPORTED_AGGS:
        raise IRCompileError(f"Unsupported aggregation '{agg}'")
    return f"{agg_norm}({inner})"


# ── Core compiler ─────────────────────────────────────────────────────────────

def compile_ir(ir: dict, allowed_joins: set) -> str:
    """
    Compile a typed IR dict to a DuckDB SQL string.
    Raises IRCompileError if any join is not in allowed_joins.
    """
    intent = ir.get("intent", "")

    if ir.get("semantic_template"):
        return _compile_semantic_template(ir, allowed_joins)

    # Specialised handler for delay and anomaly queries — these require a
    # 3-CTE pattern that the general compiler cannot express correctly.
    if intent in ("delay_analysis", "anomaly_filter"):
        return _compile_delay_anomaly(ir, allowed_joins)

    # Specialised handler for negated path queries ("never linked", "no delivery")
    if intent == "path_relation" and ir.get("negation", False):
        return _compile_path_negation(ir, allowed_joins)

    # ── Phase-2 IR extensions (issues #window, #nested, #conformance) ──────────
    # Each of these new intents has its own SQL shape that the general
    # compiler below cannot express correctly: `conformance` emits NOT EXISTS /
    # NOT IN patterns, `nested_agg` emits a correlated sub-query in WHERE,
    # `window_agg` emits window functions (OVER / PARTITION BY / ROWS BETWEEN).
    # Each specialised compiler lives below and is unit-tested in tests/.
    if intent == "conformance":
        return _compile_conformance(ir, allowed_joins)
    if intent == "nested_agg":
        return _compile_nested_agg(ir, allowed_joins)
    if intent == "window_agg":
        return _compile_window_agg(ir, allowed_joins)

    # Validate joins (hard block) for all other intents
    for j in ir.get("joins", []):
        rt = j.get("relation_type", "")
        if rt not in allowed_joins:
            raise IRCompileError(
                f"Join '{rt}' not in whitelist. "
                f"Allowed: {sorted(allowed_joins)}"
            )

    # Normalise temporal GROUP BY / ORDER BY before building clauses
    ir = _normalize_temporal(ir)

    cte_clauses   = _build_ctes(ir.get("ctes", []), allowed_joins)
    select_clause = _build_select(ir["select"])
    from_clause   = _build_from(ir["tables"], ir.get("joins", []))
    where_clause  = _build_where(ir.get("filters", []), ir.get("temporal"))
    group_clause  = _build_group(ir.get("group_by", []))
    order_clause  = _build_order(ir.get("order_by", []))
    limit_clause  = f"LIMIT {ir['limit']}" if ir.get("limit") else ""

    parts = []
    if cte_clauses:
        parts.append(f"WITH {cte_clauses}")
    parts.append(f"SELECT {select_clause}")
    parts.append(f"FROM {from_clause}")
    if where_clause:
        parts.append(f"WHERE {where_clause}")
    if group_clause:
        parts.append(f"GROUP BY {group_clause}")
    if order_clause:
        parts.append(f"ORDER BY {order_clause}")
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


# ── Semantic template compilers ───────────────────────────────────────────────

def _compile_semantic_template(ir: dict, allowed_joins: set) -> str:
    template = ir.get("semantic_template")
    if template == "orders_with_dunning_customer":
        return _compile_orders_with_dunning_customer(ir, allowed_joins)
    if template == "order_items_with_delivery_and_billing":
        return _compile_order_items_with_delivery_and_billing(ir, allowed_joins)
    if template == "billing_docs_without_payment_clearing":
        return _compile_billing_docs_without_payment_clearing(ir, allowed_joins)
    if template == "order_items_billed_not_delivered":
        return _compile_order_items_billed_not_delivered(ir, allowed_joins)
    if template == "order_items_only_order_created":
        return _compile_order_items_only_order_created(ir)
    if template == "customers_more_than_average_order_items":
        return _compile_customers_more_than_average_order_items(ir, allowed_joins)
    if template == "customers_with_delivery_events_in_period":
        return _compile_customers_with_delivery_events_in_period(ir, allowed_joins)
    raise IRCompileError(f"Unknown semantic_template '{template}'")


def _require_join(relation_type: str, allowed_joins: set) -> None:
    if relation_type not in allowed_joins:
        raise IRCompileError(f"Join '{relation_type}' not in whitelist")


def _first_alias(ir: dict, default: str) -> str:
    select_items = ir.get("select", [])
    if select_items:
        return select_items[0].get("alias") or default
    return default


def _compile_orders_with_dunning_customer(ir: dict, allowed_joins: set) -> str:
    """
    Count order items linked to a customer that has a dunning_raised event.

    This is the audited Q047 pattern. It needs an EXISTS sub-query so the
    dunning condition is tied to the linked customer, not only to the base
    order-item population.
    """
    relation_type = "order_to_customer"
    _require_join(relation_type, allowed_joins)

    alias = _first_alias(ir, "n_orders_with_dunning_customer")

    return (
        f"SELECT COUNT(DISTINCT o.object_id) AS {alias}\n"
        f"FROM objects o\n"
        f"WHERE o.object_type = 'order_item'\n"
        f"  AND EXISTS (\n"
        f"      SELECT 1\n"
        f"      FROM relations r\n"
        f"      JOIN events e ON e.object_id = r.to_object_id\n"
        f"      WHERE r.from_object_id = o.object_id\n"
        f"        AND r.relation_type = 'order_to_customer'\n"
        f"        AND e.event_type = 'dunning_raised'\n"
        f"  )"
    )


def _compile_order_items_with_delivery_and_billing(ir: dict, allowed_joins: set) -> str:
    _require_join("order_to_delivery", allowed_joins)
    _require_join("order_to_billing", allowed_joins)
    alias = _first_alias(ir, "n_order_items_with_delivery_and_billing")
    return (
        f"SELECT COUNT(DISTINCT r1.from_object_id) AS {alias}\n"
        f"FROM relations r1\n"
        f"WHERE r1.relation_type = 'order_to_delivery'\n"
        f"  AND r1.from_object_id IN (\n"
        f"      SELECT r2.from_object_id\n"
        f"      FROM relations r2\n"
        f"      WHERE r2.relation_type = 'order_to_billing'\n"
        f"  )"
    )


def _compile_billing_docs_without_payment_clearing(ir: dict, allowed_joins: set) -> str:
    _require_join("billing_to_ar", allowed_joins)
    alias = _first_alias(ir, "n_billing_docs_without_payment_clearing")
    return (
        f"SELECT COUNT(DISTINCT b.object_id) AS {alias}\n"
        f"FROM objects b\n"
        f"WHERE b.object_type = 'billing_doc'\n"
        f"  AND b.object_id NOT IN (\n"
        f"      SELECT r.from_object_id\n"
        f"      FROM relations r\n"
        f"      JOIN events e ON e.object_id = r.to_object_id\n"
        f"      WHERE r.relation_type = 'billing_to_ar'\n"
        f"        AND e.event_type = 'payment_clearing'\n"
        f"  )"
    )


def _compile_order_items_billed_not_delivered(ir: dict, allowed_joins: set) -> str:
    _require_join("order_to_billing", allowed_joins)
    _require_join("order_to_delivery", allowed_joins)
    alias = _first_alias(ir, "n_order_items_billed_not_delivered")
    return (
        f"SELECT COUNT(DISTINCT r.from_object_id) AS {alias}\n"
        f"FROM relations r\n"
        f"WHERE r.relation_type = 'order_to_billing'\n"
        f"  AND r.from_object_id NOT IN (\n"
        f"      SELECT from_object_id\n"
        f"      FROM relations\n"
        f"      WHERE relation_type = 'order_to_delivery'\n"
        f"  )"
    )


def _compile_order_items_only_order_created(ir: dict) -> str:
    alias = _first_alias(ir, "n_order_items_only_order_created")
    return (
        f"SELECT COUNT(DISTINCT object_id) AS {alias}\n"
        f"FROM (\n"
        f"    SELECT object_id\n"
        f"    FROM events\n"
        f"    WHERE object_type = 'order_item'\n"
        f"    GROUP BY object_id\n"
        f"    HAVING COUNT(DISTINCT event_type) = 1\n"
        f"       AND MAX(event_type) = 'order_created'\n"
        f") AS only_orders"
    )


def _compile_customers_more_than_average_order_items(ir: dict, allowed_joins: set) -> str:
    _require_join("order_to_customer", allowed_joins)
    alias = _first_alias(ir, "n_customers")
    return (
        f"WITH c AS (\n"
        f"    SELECT to_object_id AS customer_id, COUNT(*) AS n\n"
        f"    FROM relations\n"
        f"    WHERE relation_type = 'order_to_customer'\n"
        f"    GROUP BY to_object_id\n"
        f")\n"
        f"SELECT COUNT(*) AS {alias}\n"
        f"FROM c\n"
        f"WHERE n > (SELECT AVG(n) FROM c)"
    )


def _filter_value(ir: dict, col: str) -> Any:
    for f in ir.get("filters", []):
        if f.get("col") == col:
            return f.get("val")
    return None


def _compile_customers_with_delivery_events_in_period(ir: dict, allowed_joins: set) -> str:
    """Count customers whose linked order items have delivery events in a period."""
    _require_join("order_to_customer", allowed_joins)
    _require_join("order_to_delivery", allowed_joins)

    alias = _first_alias(ir, "n_customers_with_deliveries")
    year = _filter_value(ir, "year(timestamp)")
    month = _filter_value(ir, "month(timestamp)")

    period_filters = []
    if year is not None:
        period_filters.append(f"year(e.timestamp) = {int(year)}")
    if month is not None:
        period_filters.append(f"month(e.timestamp) = {int(month)}")
    period_sql = ""
    if period_filters:
        period_sql = "\n  AND " + "\n  AND ".join(period_filters)

    return (
        f"SELECT COUNT(DISTINCT oc.to_object_id) AS {alias}\n"
        f"FROM relations oc\n"
        f"JOIN relations od\n"
        f"  ON od.from_object_id = oc.from_object_id\n"
        f" AND od.relation_type = 'order_to_delivery'\n"
        f"JOIN events e\n"
        f"  ON e.object_id = SPLIT_PART(od.to_object_id, '_', 1)\n"
        f" AND e.object_type = 'delivery_item'\n"
        f"WHERE oc.relation_type = 'order_to_customer'\n"
        f"  AND e.event_type = 'delivery_created'"
        f"{period_sql}"
    )


def _relation_side_event_key(relation_type: str, side: str, link_col: str) -> str:
    """
    Return the event-side object key expression for a relation endpoint.

    Delivery relations are stored at delivery-item granularity
    (for example 0080011857_000010), while LIKP delivery/goods-issue events
    are stored at delivery-header granularity (for example 0080011857). The
    prefix bridge is therefore required whenever the relation endpoint is a
    delivery item and the target is an event timestamp.
    """
    if (
        relation_type == "order_to_delivery" and side == "to"
    ) or (
        relation_type == "delivery_to_billing" and side == "from"
    ):
        return f"SPLIT_PART({link_col}, '_', 1)"
    return link_col


# ── Temporal normalisation ────────────────────────────────────────────────────

def _normalize_temporal(ir: dict) -> dict:
    """
    Rewrite bare temporal alias names in group_by/order_by/select so they
    resolve to valid DuckDB expressions.

    Transforms:  group_by: ["year"]  →  group_by: ["year(timestamp)"]
    Also ensures each group_by expression appears in select.
    """
    import copy
    ir = copy.deepcopy(ir)

    # Normalise group_by
    new_group: list[str] = []
    for g in ir.get("group_by", []):
        new_group.append(_TEMPORAL_ALIAS_MAP.get(g, g))
    ir["group_by"] = new_group

    # Normalise order_by
    new_order = []
    for o in ir.get("order_by", []):
        col = _TEMPORAL_ALIAS_MAP.get(o["col"], o["col"])
        new_order.append({"col": col, "dir": o.get("dir", "ASC")})
    ir["order_by"] = new_order

    # Ensure each group_by expr is present in select
    select = ir.get("select", [])
    select_cols = {s["col"] for s in select}
    for expr in new_group:
        if expr not in select_cols:
            # Derive a clean alias: year(timestamp) → year
            raw_alias = expr.split("(")[0].replace("strftime", "period")
            select.insert(0, {"col": expr, "agg": None, "alias": raw_alias})
    ir["select"] = select

    return ir


# ── Path negation compiler ("never linked", "without") ────────────────────────

def _compile_path_negation(ir: dict, allowed_joins: set) -> str:
    """
    Compile path_relation queries that ask for objects NOT linked to another.
    E.g. "order items with no linked delivery".

    Pattern: LEFT JOIN relations ON object_id = from_object_id AND relation_type = X
             WHERE relations.from_object_id IS NULL
    """
    joins = ir.get("joins", [])
    if not joins:
        raise IRCompileError("path_relation with negation requires a join entry")
    j  = joins[0]
    rt = j.get("relation_type", "")
    if rt not in allowed_joins:
        raise IRCompileError(
            f"Join '{rt}' not in whitelist. Allowed: {sorted(allowed_joins)}"
        )

    join_col = j.get("join_col", "from_object_id")

    # SELECT clause — COUNT DISTINCT object_id by default
    select_items = ir.get("select", [])
    if select_items:
        s     = select_items[0]
        agg   = (s.get("agg") or "COUNT").upper()
        col   = s.get("col", "object_id")
        alias = s.get("alias", "n")
        dist  = s.get("distinct", True)
        inner = f"DISTINCT {col}" if dist else col
        select_expr = f"{agg}({inner}) AS {alias}"
    else:
        select_expr = "COUNT(DISTINCT o.object_id) AS n"

    # WHERE clause from object_type filter + negation IS NULL
    obj_conditions = []
    for f in ir.get("filters", []):
        col = f["col"]
        op  = f["op"].upper()
        val = f["val"]
        if isinstance(val, str):
            obj_conditions.append(f"o.{col} {op} '{val}'")
        else:
            obj_conditions.append(f"o.{col} {op} {val}")

    where_parts = obj_conditions + [f"r.{join_col} IS NULL"]
    where_clause = " AND ".join(where_parts)

    return (
        f"SELECT {select_expr}\n"
        f"FROM objects o\n"
        f"LEFT JOIN relations r\n"
        f"  ON o.object_id = r.{join_col}\n"
        f"  AND r.relation_type = '{rt}'\n"
        f"WHERE {where_clause}"
    )


# ── Delay / anomaly specialised compiler ──────────────────────────────────────

def _compile_delay_anomaly(ir: dict, allowed_joins: set) -> str:
    """
    Build delay SQL for delay_analysis / anomaly_filter.

    Two paths:
      - billing_to_ar: uses object attributes FKDAT (billing date) and AUGDT
        (clearing date) — more accurate than event timestamps for this relation.
      - all other relations: 3-CTE event-timestamp pattern.
    """
    # ── Step 1: find relation_type ──────────────────────────────────────────
    relation_type = None
    for j in ir.get("joins", []):
        rt = j.get("relation_type", "")
        if rt in allowed_joins:
            relation_type = rt
            break
    if not relation_type:
        for f in ir.get("filters", []):
            if f.get("col") == "relation_type" and f.get("val") in allowed_joins:
                relation_type = f["val"]
                break
    if not relation_type:
        raise IRCompileError(
            "delay_analysis/anomaly_filter requires a whitelisted join relation. "
            "Provide joins: [{relation_type: ...}]"
        )

    # ── Step 2: aggregation from select ────────────────────────────────────
    select_items = ir.get("select", [])
    if select_items:
        s = select_items[0]
    else:
        s = {"agg": "AVG", "alias": "avg_delay_days"}
    agg   = (s.get("agg") or "AVG").upper()
    alias = s.get("alias", "result")

    # ── Step 3: threshold for anomaly_filter ────────────────────────────────
    intent = ir.get("intent", "")
    threshold_val, threshold_op = None, ">"
    if intent == "anomaly_filter":
        for f in ir.get("filters", []):
            if f.get("col") == "delay_days":
                threshold_val = f.get("val")
                threshold_op  = f.get("op", ">")
                break

    # ── Path A: billing_to_ar — use object attributes FKDAT / AUGDT ─────────
    if relation_type == "billing_to_ar":
        return _compile_billing_to_ar_delay(s, threshold_val, threshold_op)

    # ── Path B: event-timestamp 3-CTE pattern ───────────────────────────────
    e1_etype, e1_otype, e2_etype, e2_otype = None, None, None, None
    ctes = ir.get("ctes", [])
    if len(ctes) >= 2:
        for idx, cte in enumerate(ctes[:2]):
            inner = cte.get("inner", {})
            et, ot = _extract_event_object(inner)
            if idx == 0:
                e1_etype, e1_otype = et, ot
            else:
                e2_etype, e2_otype = et, ot
    else:
        ev_filters = [f for f in ir.get("filters", []) if f.get("col") == "event_type"]
        if len(ev_filters) >= 2:
            e1_etype, e2_etype = ev_filters[0]["val"], ev_filters[1]["val"]
        elif len(ev_filters) == 1:
            e1_etype = ev_filters[0]["val"]

    if isinstance(e1_etype, list):
        e1_etype = e1_etype[0] if e1_etype else ""
    if isinstance(e2_etype, list):
        e2_etype = e2_etype[0] if e2_etype else ""
    if not e1_etype:
        e1_etype = _first_event_type_for_relation(relation_type, side="from")
    if not e2_etype:
        e2_etype = _first_event_type_for_relation(relation_type, side="to")
    if not e1_otype:
        e1_otype = _DEFAULT_OBJ_TYPE.get(e1_etype, "")
    if not e2_otype:
        e2_otype = _DEFAULT_OBJ_TYPE.get(e2_etype, "")

    e1_filter = f"event_type = '{e1_etype}'"
    if e1_otype:
        e1_filter += f" AND object_type = '{e1_otype}'"
    e2_filter = f"event_type = '{e2_etype}'"
    if e2_otype:
        e2_filter += f" AND object_type = '{e2_otype}'"

    delay_expr = "date_diff('day', e1.e1_ts, e2.e2_ts)"
    select_expr = (
        f"COUNT(*) AS {alias}" if agg == "COUNT"
        else f"{_agg_expr(agg, delay_expr, s)} AS {alias}"
    )
    threshold_clause = (
        f"\nWHERE date_diff('day', e1.e1_ts, e2.e2_ts) {threshold_op} {threshold_val}"
        if threshold_val is not None else ""
    )
    e1_join_key = _relation_side_event_key(relation_type, "from", "links.e1_id")
    e2_join_key = _relation_side_event_key(relation_type, "to", "links.e2_id")

    return (
        f"WITH e1 AS (\n"
        f"    SELECT object_id, timestamp AS e1_ts\n"
        f"    FROM events WHERE {e1_filter}\n"
        f"),\n"
        f"e2 AS (\n"
        f"    SELECT object_id, timestamp AS e2_ts\n"
        f"    FROM events WHERE {e2_filter}\n"
        f"),\n"
        f"links AS (\n"
        f"    SELECT from_object_id AS e1_id, to_object_id AS e2_id\n"
        f"    FROM relations WHERE relation_type = '{relation_type}'\n"
        f")\n"
        f"SELECT {select_expr}\n"
        f"FROM links\n"
        f"JOIN e1 ON e1.object_id = {e1_join_key}\n"
        f"JOIN e2 ON e2.object_id = {e2_join_key}"
        f"{threshold_clause}"
    )


def _compile_billing_to_ar_delay(
    select_item: dict, threshold_val: float | None, threshold_op: str
) -> str:
    """
    billing_to_ar delay uses object attributes:
      billing_doc.FKDAT (billing date) → ar_item.AUGDT (clearing date)
    More reliable than event timestamps for this relation.
    """
    agg = (select_item.get("agg") or "AVG").upper()
    alias = select_item.get("alias", "result")
    delay_expr = "date_diff('day', b.FKDAT, ar.AUGDT)"
    select_expr = (
        f"COUNT(*) AS {alias}" if agg == "COUNT"
        else f"{_agg_expr(agg, delay_expr, select_item)} AS {alias}"
    )
    threshold_clause = (
        f"\nWHERE date_diff('day', b.FKDAT, ar.AUGDT) {threshold_op} {threshold_val}"
        if threshold_val is not None else ""
    )
    return (
        f"WITH b AS (\n"
        f"    SELECT object_id, FKDAT\n"
        f"    FROM objects\n"
        f"    WHERE object_type = 'billing_doc' AND FKDAT IS NOT NULL\n"
        f"),\n"
        f"ar AS (\n"
        f"    SELECT object_id, AUGDT\n"
        f"    FROM objects\n"
        f"    WHERE object_type = 'ar_item' AND AUGDT IS NOT NULL\n"
        f"),\n"
        f"links AS (\n"
        f"    SELECT from_object_id AS bill_id, to_object_id AS ar_id\n"
        f"    FROM relations WHERE relation_type = 'billing_to_ar'\n"
        f")\n"
        f"SELECT {select_expr}\n"
        f"FROM links\n"
        f"JOIN b  ON b.object_id  = links.bill_id\n"
        f"JOIN ar ON ar.object_id = links.ar_id"
        f"{threshold_clause}"
    )


def _extract_event_object(inner_ir: dict) -> tuple[str | None, str | None]:
    """Extract (event_type, object_type) from a CTE inner IR."""
    et, ot = None, None
    for f in inner_ir.get("filters", []):
        if f.get("col") == "event_type":
            et = f["val"]
        elif f.get("col") == "object_type":
            ot = f["val"]
    return et, ot


def _first_event_type_for_relation(relation_type: str, side: str) -> str:
    """Heuristic: map relation_type → default event type for from/to side."""
    _FROM_EVENTS = {
        "billing_to_ar":    "billing_created",
        "order_to_delivery": "order_created",
        "order_to_billing": "order_created",
        "delivery_to_billing": "delivery_created",
        "order_to_customer": "order_created",
        "order_to_material": "order_created",
        "order_to_payer":   "order_created",
    }
    _TO_EVENTS = {
        "billing_to_ar":    "payment_clearing",
        "order_to_delivery": "delivery_created",
        "order_to_billing": "billing_created",
        "delivery_to_billing": "billing_created",
        "order_to_customer": "order_created",
        "order_to_material": "order_created",
        "order_to_payer":   "order_created",
    }
    if side == "from":
        return _FROM_EVENTS.get(relation_type, "")
    return _TO_EVENTS.get(relation_type, "")


def _extract_threshold(ir: dict) -> str:
    """Return a WHERE clause for an anomaly day-threshold filter."""
    for f in ir.get("filters", []):
        if f.get("col") == "delay_days":
            val = f.get("val")
            op  = f.get("op", ">")
            return f"WHERE date_diff('day', e1.e1_ts, e2.e2_ts) {op} {val}"
    return ""


# ── Clause builders ───────────────────────────────────────────────────────────

def _build_ctes(ctes: list, allowed_joins: set) -> str:
    if not ctes:
        return ""
    parts = []
    for cte in ctes:
        inner_sql = compile_ir(cte["inner"], allowed_joins)
        parts.append(f"{cte['name']} AS (\n  {inner_sql.replace(chr(10), chr(10)+'  ')}\n)")
    return ",\n".join(parts)


def _build_select(select: list) -> str:
    if not select:
        return "*"
    parts = []
    for s in select:
        col      = s["col"]
        agg      = s.get("agg")
        alias    = s.get("alias", "")
        distinct = s.get("distinct", False)
        if agg:
            inner = f"DISTINCT {col}" if distinct else col
            expr  = _agg_expr(agg, inner, s)
        else:
            expr = col
        if alias and alias != col:
            expr = f"{expr} AS {alias}"
        parts.append(expr)
    return ", ".join(parts)


def _build_from(tables: list, joins: list) -> str:
    if not tables:
        raise IRCompileError("IR has no tables")
    if len(tables) == 1 and not joins:
        return tables[0]

    primary = tables[0]
    from_str = primary

    for j in joins:
        rt         = j["relation_type"]
        from_alias = j.get("from_alias", "r_from")
        to_alias   = j.get("to_alias",   "r_to")
        join_col   = j.get("join_col",   "from_object_id")

        from_str += (
            f"\nJOIN relations AS {from_alias}_{to_alias}_rel "
            f"ON {primary}.object_id = {from_alias}_{to_alias}_rel.{join_col}"
            f"\n  AND {from_alias}_{to_alias}_rel.relation_type = '{rt}'"
        )

    return from_str


def _escape_str(val: str) -> str:
    """Escape a string literal for DuckDB: escape single quotes by doubling them."""
    return "'" + val.replace("'", "''") + "'"


def _build_where(filters: list, temporal: dict | None) -> str:
    conditions = []
    for f in filters:
        col = f["col"]
        op  = f["op"].upper()
        val = f["val"]

        if op == "IN":
            vals = ", ".join(_escape_str(v) if isinstance(v, str) else str(v) for v in val)
            conditions.append(f"{col} IN ({vals})")
        elif op == "BETWEEN":
            lo, hi = val
            lo_s = _escape_str(lo) if isinstance(lo, str) else str(lo)
            hi_s = _escape_str(hi) if isinstance(hi, str) else str(hi)
            conditions.append(f"{col} BETWEEN {lo_s} AND {hi_s}")
        elif isinstance(val, str):
            conditions.append(f"{col} {op} {_escape_str(val)}")
        elif val is None:
            conditions.append(f"{col} IS NULL")
        else:
            conditions.append(f"{col} {op} {val}")

    if temporal:
        col = temporal["col"]
        if temporal.get("year") is not None:
            conditions.append(f"year({col}) = {temporal['year']}")
        if temporal.get("month") is not None:
            conditions.append(f"month({col}) = {temporal['month']}")

    return " AND ".join(conditions)


def _build_group(group_by: list) -> str:
    return ", ".join(group_by)


def _build_order(order_by: list) -> str:
    parts = []
    for o in order_by:
        direction = o.get("dir", "ASC").upper()
        parts.append(f"{o['col']} {direction}")
    return ", ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 extensions: conformance / nested_agg / window_agg
# ──────────────────────────────────────────────────────────────────────────────
#
# These three intents extend the six core intents (count_filter, group_topk,
# temporal_trend, path_relation, delay_analysis, anomaly_filter) to cover
# OCEL analytical questions (missing events, above-average filters, rolling
# aggregates) that require dedicated IR shapes. Each has its own compiler;
# the general compiler path above is not touched.
#
# IR schema extensions (each field below is optional on intents that do not
# need it):
#
#   intent = "conformance":
#     {
#       "intent":      "conformance",
#       "check_type":  "missing_event" | "missing_relation" |
#                      "set_difference_relations" | "attribute_null",
#       "base":        {"object_type": "order_item"},   # the main population
#       "missing_event_type":      "goods_issue",       # for missing_event
#       "missing_relation_type":   "order_to_delivery", # for missing_relation
#       "also_missing_relation":   "order_to_billing",  # for set_difference
#       "has_relation_type":       "order_to_billing",  # for set_difference
#       "attribute_null_column":   "ZTERM",             # for attribute_null
#       "attribute_null_logic":    "or"                 # "or"|"and" for multi-col
#       "extra_null_columns":      ["WAERK"],
#       "select":      [{"agg": "COUNT", "col": "object_id",
#                        "distinct": true, "alias": "n"}],
#       "filters":     [...]                            # extra WHERE on base
#     }
#
#   intent = "nested_agg":
#     {
#       "intent":              "nested_agg",
#       "base":                {"table": "events", "group_by": "event_type"},
#       "metric":              {"agg": "COUNT", "col": "*"},
#       "comparison":          {"op": ">", "reference": "AVG"},
#           # reference in {AVG, MEDIAN, MAX, MIN, REFERENCE_VALUE}
#       "reference_value":     2004,                    # only for REFERENCE_VALUE
#       "reference_group_key": "event_type",            # only for REFERENCE_VALUE
#       "filters":             [...]                    # optional pre-aggregation
#     }
#
#   intent = "window_agg":
#     {
#       "intent":    "window_agg",
#       "base":      {"table": "events", "partition_by": null, "order_by": "year"},
#       "metric":    {"agg": "COUNT", "col": "*", "alias": "n"},
#       "window":    {
#           "function": "AVG" | "SUM" | "LAG" | "RANK" |
#                       "DENSE_RANK" | "ROW_NUMBER" | "PERCENT_RANK",
#           "partition_by": null | str,
#           "order_by":     str,
#           "rows_preceding": 2,   # for moving average / running total
#           "offset":         1,   # for LAG / LEAD
#           "alias":          "moving_avg"
#       },
#       "filters":   [...],
#       "group_by":  [str, ...],   # outer GROUP BY (before window)
#       "limit":     int | null
#     }


def _compile_conformance(ir: dict, allowed_joins: set) -> str:
    """Emit NOT EXISTS / NOT IN / attribute-null SQL for conformance checks.

    The four supported check_types and their SQL shapes:
      - missing_event           : base.object_id NOT IN (objects that HAVE the event)
      - missing_relation        : base.object_id NOT IN (from_object_id of the relation)
      - set_difference_relations: has relation A AND NOT has relation B
      - attribute_null          : base.object_type + <col> IS NULL (+ optional OR/AND chain)

    All patterns use a main `FROM objects o` query with a correlated NOT IN
    sub-query, which DuckDB plans as an anti-join.
    """
    check_type = ir.get("check_type", "")
    base       = ir.get("base", {})
    base_otype = base.get("object_type", "")
    if not base_otype:
        raise IRCompileError(
            "conformance requires base.object_type (e.g. 'order_item')"
        )

    # Build the standard SELECT expression — defaults to COUNT(DISTINCT object_id).
    select_items = ir.get("select", [])
    if select_items:
        s = select_items[0]
        agg   = (s.get("agg") or "COUNT").upper()
        col   = s.get("col", "object_id")
        alias = s.get("alias", "n")
        inner = f"DISTINCT {col}" if s.get("distinct", True) else col
        select_expr = f"{agg}({inner}) AS {alias}"
    else:
        select_expr = "COUNT(DISTINCT o.object_id) AS n"

    # Extra filters on the base set (rare — e.g. WAERK='USD').
    extra_filters = []
    for f in ir.get("filters", []):
        col = f["col"]
        op  = f["op"].upper()
        val = f["val"]
        if isinstance(val, str):
            extra_filters.append(f"o.{col} {op} {_escape_str(val)}")
        else:
            extra_filters.append(f"o.{col} {op} {val}")
    extra_where = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""

    # ── Branch A: missing_event ─────────────────────────────────────────────
    if check_type == "missing_event":
        ev = ir.get("missing_event_type")
        if not ev:
            raise IRCompileError("missing_event requires missing_event_type")
        return (
            f"SELECT {select_expr}\n"
            f"FROM objects o\n"
            f"WHERE o.object_type = {_escape_str(base_otype)}{extra_where}\n"
            f"  AND o.object_id NOT IN (\n"
            f"      SELECT object_id FROM events WHERE event_type = {_escape_str(ev)}\n"
            f"  )"
        )

    # ── Branch B: missing_relation ──────────────────────────────────────────
    if check_type == "missing_relation":
        rt = ir.get("missing_relation_type")
        if not rt:
            raise IRCompileError("missing_relation requires missing_relation_type")
        if rt not in allowed_joins:
            raise IRCompileError(f"Relation '{rt}' not in whitelist")
        return (
            f"SELECT {select_expr}\n"
            f"FROM objects o\n"
            f"WHERE o.object_type = {_escape_str(base_otype)}{extra_where}\n"
            f"  AND o.object_id NOT IN (\n"
            f"      SELECT from_object_id FROM relations WHERE relation_type = {_escape_str(rt)}\n"
            f"  )"
        )

    # ── Branch C: set_difference_relations (has A but not B) ────────────────
    if check_type == "set_difference_relations":
        has_rt     = ir.get("has_relation_type")
        missing_rt = ir.get("also_missing_relation")
        if not has_rt or not missing_rt:
            raise IRCompileError(
                "set_difference_relations requires has_relation_type + also_missing_relation"
            )
        if has_rt not in allowed_joins or missing_rt not in allowed_joins:
            raise IRCompileError(f"Relation '{has_rt}' or '{missing_rt}' not in whitelist")
        # Here the main FROM is the relations table because the "base set" is
        # already defined by having relation A.
        return (
            f"SELECT COUNT(DISTINCT from_object_id) AS n\n"
            f"FROM relations r1\n"
            f"WHERE r1.relation_type = {_escape_str(has_rt)}\n"
            f"  AND r1.from_object_id NOT IN (\n"
            f"      SELECT from_object_id FROM relations WHERE relation_type = {_escape_str(missing_rt)}\n"
            f"  )"
        )

    # ── Branch D: attribute_null ────────────────────────────────────────────
    if check_type == "attribute_null":
        col = ir.get("attribute_null_column")
        if not col:
            raise IRCompileError("attribute_null requires attribute_null_column")
        extras = ir.get("extra_null_columns", []) or []
        logic  = (ir.get("attribute_null_logic") or "and").upper()
        if logic not in ("AND", "OR"):
            raise IRCompileError("attribute_null_logic must be 'and' or 'or'")
        null_terms = [f"o.{col} IS NULL"] + [f"o.{c} IS NULL" for c in extras]
        null_expr  = f" {logic} ".join(null_terms)
        if len(null_terms) > 1:
            null_expr = f"({null_expr})"
        return (
            f"SELECT {select_expr}\n"
            f"FROM objects o\n"
            f"WHERE o.object_type = {_escape_str(base_otype)}{extra_where}\n"
            f"  AND {null_expr}"
        )

    raise IRCompileError(
        f"Unknown conformance check_type '{check_type}'. Must be one of "
        "missing_event | missing_relation | set_difference_relations | attribute_null"
    )


def _compile_nested_agg(ir: dict, allowed_joins: set) -> str:
    """Emit a correlated-sub-query filter: 'which X have metric > AVG(metric)?'.

    The IR describes a base aggregate (e.g. per event_type count) and a
    comparison against a population statistic (AVG, MEDIAN, MAX, MIN) of the
    same aggregate. We compile to a CTE + a correlated sub-query in the outer
    WHERE clause — the shape Nan et al. 2023's "above-average" questions expect.

    Example IR:
      {"intent": "nested_agg",
       "base":   {"table": "events", "group_by": "event_type"},
       "metric": {"agg": "COUNT", "col": "*"},
       "comparison": {"op": ">", "reference": "AVG"}}
    produces:
      WITH t AS (SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type)
      SELECT event_type, n FROM t WHERE n > (SELECT AVG(n) FROM t)
    """
    base = ir.get("base", {})
    table = base.get("table", "events")
    group_expr = base.get("group_by")
    if not group_expr:
        raise IRCompileError("nested_agg requires base.group_by (e.g. 'event_type')")

    # Always alias the group expression in the CTE so subsequent references
    # don't have to repeat the function-call syntax. For a bare column
    # ('event_type') the alias is the column itself; for an expression
    # ('year(timestamp)') we generate a stable alias 'gkey'. Outer SELECT
    # and the comparison RHS both reference this alias.
    if "(" in group_expr or " " in group_expr:
        group_alias = "gkey"
        group_select = f"{group_expr} AS {group_alias}"
    else:
        group_alias = group_expr
        group_select = group_expr

    metric = ir.get("metric", {})
    agg = (metric.get("agg") or "COUNT").upper()
    col = metric.get("col") or "*"
    metric_alias = "n"  # stable alias the outer WHERE references

    # Pre-aggregation filters go in the CTE's WHERE clause.
    cte_filters = []
    for f in ir.get("filters", []):
        c, op, val = f["col"], f["op"].upper(), f["val"]
        if isinstance(val, str):
            cte_filters.append(f"{c} {op} {_escape_str(val)}")
        else:
            cte_filters.append(f"{c} {op} {val}")
    cte_where = ("WHERE " + " AND ".join(cte_filters)) if cte_filters else ""

    cte = (
        f"WITH t AS (SELECT {group_select}, {agg}({col}) AS {metric_alias} "
        f"FROM {table} {cte_where} GROUP BY {group_expr})"
    ).replace("  ", " ").strip()

    comparison = ir.get("comparison", {})
    op        = comparison.get("op", ">")
    reference = (comparison.get("reference") or "AVG").upper()

    # Population-statistic reference vs a literal reference row value.
    if reference in ("AVG", "MEDIAN", "MAX", "MIN"):
        # DuckDB uses MEDIAN as an aggregate directly.
        rhs = f"(SELECT {reference}({metric_alias}) FROM t)"
    elif reference == "REFERENCE_VALUE":
        # "Years that had more X events than 2004" — reference a specific row.
        # The reference is matched against the CTE's group alias, NOT the raw
        # expression, because the CTE has already aliased the expression.
        ref_val = ir.get("reference_value")
        if ref_val is None:
            raise IRCompileError(
                "REFERENCE_VALUE requires a 'reference_value' field"
            )
        rhs_val = _escape_str(ref_val) if isinstance(ref_val, str) else ref_val
        rhs = f"(SELECT {metric_alias} FROM t WHERE {group_alias} = {rhs_val})"
    else:
        raise IRCompileError(
            f"Unknown comparison.reference '{reference}'. "
            "Must be AVG | MEDIAN | MAX | MIN | REFERENCE_VALUE"
        )

    # Optional outer SELECT override — defaults to (group_alias, metric_alias).
    select_expr = f"{group_alias}, {metric_alias}"
    if ir.get("select"):
        # Caller may want COUNT(*) of rows that pass — supports "how many X > avg?".
        s = ir["select"][0]
        outer_agg = (s.get("agg") or "").upper()
        if outer_agg:
            a = s.get("alias", "n")
            select_expr = f"{outer_agg}(*) AS {a}"

    # If the IR specifies order_by referencing the group expression, rewrite
    # it to use the alias so the outer SELECT can resolve it.
    order_clause = ""
    if ir.get("order_by"):
        rewritten = []
        for o in ir["order_by"]:
            o_col = o.get("col")
            if o_col == group_expr and group_alias != group_expr:
                rewritten.append({**o, "col": group_alias})
            else:
                rewritten.append(o)
        order_clause = "\nORDER BY " + _build_order(rewritten)

    return (
        f"{cte}\n"
        f"SELECT {select_expr} FROM t\n"
        f"WHERE {metric_alias} {op} {rhs}"
        f"{order_clause}"
    )


def _compile_window_agg(ir: dict, allowed_joins: set) -> str:
    """Emit SQL with a window function (OVER / PARTITION BY / ORDER BY / frame).

    IR carries both a base GROUP BY aggregate and a `window` spec describing
    the secondary window computation. Output has two stages:
      1. a CTE that produces the grouped metric
      2. an outer SELECT that layers the window function over the CTE

    Supported `window.function` values:
      - AVG / SUM                             : rolling / cumulative (uses rows_preceding)
      - LAG / LEAD                            : offset-based comparisons
      - RANK / DENSE_RANK / ROW_NUMBER        : ordering-based rankings
      - PERCENT_RANK                          : normalised rank
    """
    base = ir.get("base", {})
    table = base.get("table", "events")
    partition_key  = base.get("partition_by")  # optional GROUP BY dim outside window
    outer_key      = base.get("order_by")       # primary ordering column (usually temporal)
    if not outer_key:
        raise IRCompileError("window_agg requires base.order_by (e.g. 'year' or 'month')")

    metric = ir.get("metric", {})
    metric_agg  = (metric.get("agg") or "COUNT").upper()
    metric_col  = metric.get("col") or "*"
    metric_alias = metric.get("alias") or "n"

    # ── Build the CTE that produces (outer_key, partition_key, metric) ───────
    cte_filters = []
    for f in ir.get("filters", []):
        c, op, val = f["col"], f["op"].upper(), f["val"]
        if isinstance(val, str):
            cte_filters.append(f"{c} {op} {_escape_str(val)}")
        else:
            cte_filters.append(f"{c} {op} {val}")
    # temporal_alias expansion — users often say order_by: "month" to mean MONTH(timestamp).
    outer_key_expr = _TEMPORAL_ALIAS_MAP.get(outer_key, outer_key)
    group_cols = [f"{outer_key_expr} AS {outer_key}"]
    if partition_key:
        partition_expr = _TEMPORAL_ALIAS_MAP.get(partition_key, partition_key)
        group_cols.insert(0, f"{partition_expr} AS {partition_key}")
    cte_select = ", ".join(group_cols + [f"{metric_agg}({metric_col}) AS {metric_alias}"])
    cte_group = ", ".join([outer_key] if not partition_key else [partition_key, outer_key])
    cte_group_expr = ", ".join(
        [outer_key_expr] if not partition_key else [partition_expr, outer_key_expr]
    )
    cte_where = ("WHERE " + " AND ".join(cte_filters)) if cte_filters else ""
    cte = (
        f"WITH t AS (\n"
        f"  SELECT {cte_select}\n"
        f"  FROM {table} {cte_where}\n"
        f"  GROUP BY {cte_group_expr}\n"
        f")"
    ).replace("  ", " ", 1)

    # ── Build the window function expression ─────────────────────────────────
    window = ir.get("window", {})
    wfunc  = (window.get("function") or "AVG").upper()
    wpart  = window.get("partition_by") or partition_key
    worder = window.get("order_by") or outer_key
    walias = window.get("alias", f"{wfunc.lower()}_{metric_alias}")
    rows_preceding = window.get("rows_preceding")
    offset         = window.get("offset", 1)

    over_parts = []
    if wpart:
        over_parts.append(f"PARTITION BY {wpart}")
    over_parts.append(f"ORDER BY {worder}")
    if wfunc in ("AVG", "SUM") and rows_preceding is not None:
        over_parts.append(f"ROWS BETWEEN {int(rows_preceding)} PRECEDING AND CURRENT ROW")
    over_clause = " ".join(over_parts)

    if wfunc in ("LAG", "LEAD"):
        window_expr = f"{wfunc}({metric_alias}, {int(offset)}) OVER ({over_clause})"
    elif wfunc in ("RANK", "DENSE_RANK", "ROW_NUMBER", "PERCENT_RANK"):
        # Rank functions ignore the metric column — they only need ordering.
        # Override ORDER BY if caller specified it via window.order_by.
        window_expr = f"{wfunc}() OVER ({over_clause})"
    else:  # AVG / SUM (rolling / cumulative)
        window_expr = f"{wfunc}({metric_alias}) OVER ({over_clause})"

    # ── Outer SELECT ────────────────────────────────────────────────────────
    outer_cols = [c.split(" AS ")[-1] for c in group_cols]  # just the aliases
    outer_cols.append(metric_alias)
    outer_cols.append(f"{window_expr} AS {walias}")
    outer_select = ", ".join(outer_cols)

    order_by = ir.get("order_by") or [{"col": outer_key, "dir": "ASC"}]
    order_clause = "ORDER BY " + _build_order(order_by)

    limit_clause = f"LIMIT {ir['limit']}" if ir.get("limit") else ""

    sql = f"{cte}\nSELECT {outer_select}\nFROM t\n{order_clause}"
    if limit_clause:
        sql += f"\n{limit_clause}"
    return sql


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wl = {"billing_to_ar", "order_to_delivery", "order_to_customer",
          "order_to_material", "order_to_billing", "delivery_to_billing", "order_to_payer"}

    # Test 1: simple count filter
    ir1 = {
        "intent":   "count_filter",
        "tables":   ["events"],
        "select":   [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters":  [{"table": "events", "col": "event_type", "op": "=", "val": "billing_created"}],
        "joins":    [], "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }
    print("Test 1 — count_filter:")
    print(compile_ir(ir1, wl))

    # Test 2: temporal trend (bare "year" group_by — common LLM mistake)
    ir2 = {
        "intent":   "temporal_trend",
        "tables":   ["events"],
        "select":   [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters":  [{"table": "events", "col": "event_type", "op": "=", "val": "order_created"}],
        "joins":    [], "group_by": ["year"],
        "order_by": [{"col": "year", "dir": "ASC"}],
        "limit": None, "temporal": None, "ctes": [],
    }
    print("\nTest 2 — temporal_trend (bare 'year' normalization):")
    print(compile_ir(ir2, wl))

    # Test 3: delay_analysis
    ir3 = {
        "intent":   "delay_analysis",
        "tables":   ["events"],
        "select":   [{"col": "*", "agg": "AVG", "alias": "avg_days"}],
        "filters":  [],
        "joins":    [{"relation_type": "billing_to_ar", "from_alias": "bill", "to_alias": "ar", "join_col": "from_object_id"}],
        "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }
    print("\nTest 3 — delay_analysis:")
    print(compile_ir(ir3, wl))

    # Test 4: anomaly_filter with threshold
    ir4 = {
        "intent":   "anomaly_filter",
        "tables":   ["events"],
        "select":   [{"col": "*", "agg": "COUNT", "alias": "n_anomalous"}],
        "filters":  [{"col": "delay_days", "op": ">", "val": 873}],
        "joins":    [{"relation_type": "billing_to_ar", "from_alias": "bill", "to_alias": "ar", "join_col": "from_object_id"}],
        "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }
    print("\nTest 4 — anomaly_filter (threshold 873 days):")
    print(compile_ir(ir4, wl))

    # Test 5: path negation ("order items with no linked delivery")
    ir5_neg = {
        "intent":   "path_relation",
        "negation": True,
        "tables":   ["objects"],
        "select":   [{"col": "object_id", "agg": "COUNT", "alias": "n", "distinct": True}],
        "filters":  [{"table": "objects", "col": "object_type", "op": "=", "val": "order_item"}],
        "joins":    [{"relation_type": "order_to_delivery", "from_alias": "ord", "to_alias": "del", "join_col": "from_object_id"}],
        "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
    }
    print("\nTest 5 — path_negation (LEFT JOIN IS NULL):")
    print(compile_ir(ir5_neg, wl))

    # Test 6: invalid join — should raise
    ir5 = {
        "intent": "path_relation", "tables": ["events"],
        "select": [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters": [], "group_by": [], "order_by": [], "limit": None, "temporal": None, "ctes": [],
        "joins": [{"relation_type": "events_to_fake", "from_alias": "x", "to_alias": "y", "join_col": "from_object_id"}],
    }
    print("\nTest 6 — invalid join (should raise):")
    try:
        compile_ir(ir5, wl)
    except IRCompileError as e:
        print(f"IRCompileError caught: {e}")
