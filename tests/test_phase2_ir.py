"""
test_phase2_ir.py — unit tests for the three Phase-2 IR intents.

Covers:
  - conformance.missing_event
  - conformance.missing_relation
  - conformance.set_difference_relations
  - conformance.attribute_null (single + multi-column OR)
  - nested_agg with AVG / MEDIAN / REFERENCE_VALUE comparisons
  - window_agg with AVG (rolling), SUM (cumulative), LAG, RANK, PARTITION BY

Tests validate that compile_ir emits SQL containing the expected clauses
(not executing against DuckDB — that is done via the benchmark run). The
verifier tests confirm the new intents pass structural validation.
"""

from __future__ import annotations

import pytest

from nl2ocel.ir_to_sql import compile_ir, IRCompileError


ALLOWED_JOINS = {
    "billing_to_ar", "order_to_delivery", "order_to_customer",
    "order_to_material", "order_to_billing", "delivery_to_billing",
    "order_to_payer",
}


# ─────────────────────────────────────────────────────────────────────────────
# conformance
# ─────────────────────────────────────────────────────────────────────────────

def test_conformance_missing_event_emits_not_in():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "missing_event",
        "base": {"object_type": "order_item"},
        "missing_event_type": "goods_issue",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "FROM objects o" in sql
    assert "o.object_type = 'order_item'" in sql
    assert "NOT IN" in sql
    assert "event_type = 'goods_issue'" in sql


def test_conformance_missing_relation_checks_whitelist():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "missing_relation",
        "base": {"object_type": "delivery_item"},
        "missing_relation_type": "delivery_to_billing",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "relation_type = 'delivery_to_billing'" in sql
    assert "NOT IN" in sql


def test_conformance_rejects_unwhitelisted_relation():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "missing_relation",
        "base": {"object_type": "order_item"},
        "missing_relation_type": "fake_relation",
    }
    with pytest.raises(IRCompileError, match="not in whitelist"):
        compile_ir(ir, ALLOWED_JOINS)


def test_conformance_set_difference_pair_of_relations():
    ir = {
        "intent": "conformance",
        "tables": ["relations"], "select": [],
        "check_type": "set_difference_relations",
        "base": {"object_type": "order_item"},
        "has_relation_type": "order_to_billing",
        "also_missing_relation": "order_to_delivery",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "relations r1" in sql
    assert "r1.relation_type = 'order_to_billing'" in sql
    assert "NOT IN" in sql
    assert "relation_type = 'order_to_delivery'" in sql


def test_conformance_attribute_null_single_col():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "attribute_null",
        "base": {"object_type": "billing_doc"},
        "attribute_null_column": "ZTERM",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "o.object_type = 'billing_doc'" in sql
    assert "o.ZTERM IS NULL" in sql


def test_conformance_attribute_null_or_chain():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "attribute_null",
        "base": {"object_type": "billing_doc"},
        "attribute_null_column": "NETWR",
        "extra_null_columns": ["WAERK"],
        "attribute_null_logic": "or",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "o.NETWR IS NULL OR o.WAERK IS NULL" in sql


def test_conformance_missing_event_requires_event_type():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "missing_event",
        "base": {"object_type": "order_item"},
    }
    with pytest.raises(IRCompileError, match="missing_event_type"):
        compile_ir(ir, ALLOWED_JOINS)


def test_conformance_rejects_unknown_check_type():
    ir = {
        "intent": "conformance",
        "tables": ["objects"], "select": [],
        "check_type": "fake_check",
        "base": {"object_type": "order_item"},
    }
    with pytest.raises(IRCompileError, match="Unknown conformance check_type"):
        compile_ir(ir, ALLOWED_JOINS)


# ─────────────────────────────────────────────────────────────────────────────
# nested_agg
# ─────────────────────────────────────────────────────────────────────────────

def test_nested_agg_avg_reference():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "group_by": "event_type"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">", "reference": "AVG"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "WITH t AS" in sql
    assert "GROUP BY event_type" in sql
    assert "n > (SELECT AVG(n) FROM t)" in sql


def test_nested_agg_median_reference():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "group_by": "event_type"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">=", "reference": "MEDIAN"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "n >= (SELECT MEDIAN(n) FROM t)" in sql


def test_nested_agg_reference_value():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "group_by": "year(timestamp)"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">", "reference": "REFERENCE_VALUE"},
        "reference_value": 2004,
        "reference_group_key": "year(timestamp)",
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    # Compiler aliases expression-form group_by to a stable `gkey` so the
    # outer SELECT and the REFERENCE_VALUE comparison resolve cleanly.
    assert "year(timestamp) AS gkey" in sql
    assert "WHERE gkey = 2004" in sql


def test_nested_agg_count_of_pass_rows():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [{"agg": "COUNT", "alias": "n"}],
        "base":   {"table": "events", "group_by": "event_type"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">", "reference": "AVG"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "SELECT COUNT(*) AS n FROM t" in sql


def test_nested_agg_requires_group_by():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">", "reference": "AVG"},
    }
    with pytest.raises(IRCompileError, match="base.group_by"):
        compile_ir(ir, ALLOWED_JOINS)


def test_nested_agg_rejects_unknown_reference():
    ir = {
        "intent": "nested_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "group_by": "event_type"},
        "metric": {"agg": "COUNT", "col": "*"},
        "comparison": {"op": ">", "reference": "QUARTILE"},
    }
    with pytest.raises(IRCompileError, match="Unknown comparison.reference"):
        compile_ir(ir, ALLOWED_JOINS)


# ─────────────────────────────────────────────────────────────────────────────
# window_agg
# ─────────────────────────────────────────────────────────────────────────────

def test_window_agg_rolling_moving_average():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "order_by": "month"},
        "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
        "window": {"function": "AVG", "order_by": "month",
                   "rows_preceding": 2, "alias": "moving_avg"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "AVG(n) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg" in sql


def test_window_agg_cumulative_sum():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "order_by": "year"},
        "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
        "window": {"function": "SUM", "order_by": "year", "alias": "cumulative"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "SUM(n) OVER (ORDER BY year) AS cumulative" in sql


def test_window_agg_lag_with_offset():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "order_by": "year"},
        "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
        "window": {"function": "LAG", "order_by": "year", "offset": 1, "alias": "prev_n"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "LAG(n, 1) OVER (ORDER BY year) AS prev_n" in sql


def test_window_agg_rank_function_ignores_metric():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "order_by": "event_type"},
        "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
        "window": {"function": "RANK", "order_by": "n", "alias": "rank"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "RANK() OVER (ORDER BY n) AS rank" in sql


def test_window_agg_partition_by():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events", "partition_by": "object_type", "order_by": "year"},
        "metric": {"agg": "COUNT", "col": "*", "alias": "n"},
        "window": {"function": "ROW_NUMBER", "partition_by": "object_type",
                   "order_by": "n DESC", "alias": "rn"},
    }
    sql = compile_ir(ir, ALLOWED_JOINS)
    assert "PARTITION BY object_type" in sql
    assert "ROW_NUMBER() OVER" in sql


def test_window_agg_requires_order_by():
    ir = {
        "intent": "window_agg",
        "tables": ["events"], "select": [],
        "base":   {"table": "events"},
        "metric": {"agg": "COUNT", "col": "*"},
        "window": {"function": "AVG"},
    }
    with pytest.raises(IRCompileError, match="order_by"):
        compile_ir(ir, ALLOWED_JOINS)
