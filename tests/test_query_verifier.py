"""Unit tests for query_verifier.py — accept / reject / repair verdicts."""

import pytest

from nl2ocel.baseline_translator import check_join_hallucination
from nl2ocel.pipeline import _check_sql_policy
from nl2ocel.query_verifier import verify_ir, VerifyResult

SCHEMA = {
    "events":    {"event_id", "event_type", "timestamp", "object_id", "object_type"},
    "objects":   {"object_id", "object_type", "FKDAT", "AUGDT", "WAERK"},
    "relations": {"from_object_id", "to_object_id", "relation_type",
                  "from_object_type", "to_object_type"},
}
JOINS = {
    "billing_to_ar", "order_to_delivery", "order_to_billing",
    "delivery_to_billing", "order_to_customer", "order_to_material", "order_to_payer",
}
ENUMS = {
    "event_type":    {"billing_created", "order_created", "delivery_created",
                      "payment_clearing", "goods_receipt"},
    "object_type":   {"billing_doc", "order_item", "delivery_item",
                      "ar_item", "customer", "material"},
    "relation_type": set(JOINS),
}


def _good_ir(**kwargs):
    base = {
        "intent": "count_filter", "tables": ["events"],
        "select": [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters": [], "joins": [], "group_by": [], "order_by": [],
        "limit": None, "temporal": None, "ctes": [],
    }
    base.update(kwargs)
    return base


# ── accept cases ──────────────────────────────────────────────────────────────

def test_simple_count_accepts():
    r = verify_ir(_good_ir(), SCHEMA, JOINS)
    assert r.ok


def test_valid_event_type_filter_accepts():
    ir = _good_ir(filters=[
        {"table": "events", "col": "event_type", "op": "=", "val": "billing_created"}
    ])
    r = verify_ir(ir, SCHEMA, JOINS, enum_values=ENUMS)
    assert r.ok


def test_valid_join_accepts():
    ir = _good_ir(
        intent="path_relation",
        joins=[{"relation_type": "order_to_delivery", "from_alias": "o",
                "to_alias": "d", "join_col": "from_object_id"}],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.ok


def test_year_expression_in_select_accepts():
    ir = _good_ir(
        intent="temporal_trend",
        select=[{"col": "year(timestamp)", "agg": None, "alias": "yr"},
                {"col": "*", "agg": "COUNT", "alias": "n"}],
        group_by=["year(timestamp)"],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.ok


# ── reject cases ──────────────────────────────────────────────────────────────

def test_missing_intent_rejects():
    ir = {"tables": ["events"], "select": [{"col": "*", "agg": "COUNT", "alias": "n"}]}
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.status == "reject"
    assert any("Missing required IR key" in e for e in r.errors)


def test_unknown_intent_errors():
    ir = _good_ir(intent="magic_query")
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("Unknown intent" in e for e in r.errors)


def test_unknown_table_errors():
    ir = _good_ir(tables=["events", "ghost_table"])
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("ghost_table" in e for e in r.errors)


def test_hallucinated_column_errors():
    ir = _good_ir(select=[{"col": "fake_col", "agg": None, "alias": "x"}])
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("fake_col" in e for e in r.errors)


def test_invalid_join_errors():
    ir = _good_ir(
        intent="path_relation",
        joins=[{"relation_type": "events_to_nowhere", "from_alias": "a",
                "to_alias": "b", "join_col": "from_object_id"}],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("not in whitelist" in e for e in r.errors)


def test_bad_enum_value_errors():
    ir = _good_ir(filters=[
        {"table": "events", "col": "event_type", "op": "=", "val": "nonexistent_type"}
    ])
    r = verify_ir(ir, SCHEMA, JOINS, enum_values=ENUMS)
    assert not r.ok


def test_sql_policy_allowed_tables_enforced():
    ir = _good_ir(tables=["objects"])
    policy = {"allowed_tables": ["events"]}
    r = verify_ir(ir, SCHEMA, JOINS, policy=policy)
    assert not r.ok
    assert any("not allowed by SQL policy" in e for e in r.errors)


def test_sql_policy_allowed_predicates_enforced():
    ir = _good_ir(filters=[
        {"table": "events", "col": "event_type", "op": "LIKE", "val": "%billing%"}
    ])
    policy = {"allowed_predicates": ["="]}
    r = verify_ir(ir, SCHEMA, JOINS, policy=policy)
    assert not r.ok
    assert any("Filter op 'LIKE'" in e for e in r.errors)


# ── anomaly_filter specific ───────────────────────────────────────────────────

def test_anomaly_filter_missing_threshold_errors():
    ir = _good_ir(
        intent="anomaly_filter",
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
        filters=[],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("threshold" in e for e in r.errors)


def test_anomaly_filter_missing_join_errors():
    ir = _good_ir(
        intent="anomaly_filter",
        filters=[{"col": "delay_days", "op": ">", "val": 100}],
        joins=[],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert not r.ok
    assert any("join" in e.lower() for e in r.errors)


def test_anomaly_filter_valid_accepts():
    ir = _good_ir(
        intent="anomaly_filter",
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
        filters=[{"col": "delay_days", "op": ">", "val": 873}],
        select=[{"col": "*", "agg": "COUNT", "alias": "n"}],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.ok


# ── repair classification ─────────────────────────────────────────────────────

def test_bad_join_classified_as_repair_not_reject():
    ir = _good_ir(
        intent="path_relation",
        joins=[{"relation_type": "wrong_join", "from_alias": "a",
                "to_alias": "b", "join_col": "from_object_id"}],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.status == "repair"
    assert "allowed_joins" in r.repair_hint


def test_repair_hint_contains_allowed_joins():
    ir = _good_ir(
        intent="path_relation",
        joins=[{"relation_type": "bad_rel", "from_alias": "x",
                "to_alias": "y", "join_col": "from_object_id"}],
    )
    r = verify_ir(ir, SCHEMA, JOINS)
    assert r.repair_hint.get("allowed_joins") is not None
    assert "billing_to_ar" in r.repair_hint["allowed_joins"]


def test_baseline_join_hallucination_detects_in_lists():
    sql = "SELECT * FROM relations WHERE relation_type IN ('order_to_delivery', 'fake_join')"
    assert check_join_hallucination(sql, set(JOINS))


def test_sql_policy_enforces_allowed_predicates():
    policy = {
        "blocked_sql_keywords": [],
        "allowed_tables": ["events"],
        "allowed_aggregations": ["COUNT"],
        "allowed_predicates": ["="],
    }
    sql = "SELECT COUNT(*) FROM events WHERE year(timestamp) > 2000"
    assert _check_sql_policy(sql, policy) == "SQL uses predicate '>' outside allowed_predicates"
