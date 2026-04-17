"""Unit tests for ir_to_sql.py — IR compiler correctness."""

import pytest

from nl2ocel.ir_to_sql import compile_ir, IRCompileError

WL = {
    "billing_to_ar", "order_to_delivery", "order_to_billing",
    "delivery_to_billing", "order_to_customer", "order_to_material", "order_to_payer",
}


def _ir(**kwargs):
    base = {
        "intent": "count_filter", "tables": ["events"],
        "select": [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters": [], "joins": [], "group_by": [], "order_by": [],
        "limit": None, "temporal": None, "ctes": [],
    }
    base.update(kwargs)
    return base


# ── count_filter ──────────────────────────────────────────────────────────────

def test_count_filter_simple():
    sql = compile_ir(_ir(filters=[
        {"table": "events", "col": "event_type", "op": "=", "val": "billing_created"}
    ]), WL)
    assert "COUNT(*)" in sql
    assert "event_type = 'billing_created'" in sql


def test_count_filter_temporal_year():
    sql = compile_ir(_ir(temporal={"table": "events", "col": "timestamp", "year": 2019, "month": None}), WL)
    assert "year(timestamp) = 2019" in sql


def test_count_filter_in_operator():
    sql = compile_ir(_ir(filters=[
        {"table": "events", "col": "event_type", "op": "IN",
         "val": ["billing_created", "order_created"]}
    ]), WL)
    assert "IN ('billing_created', 'order_created')" in sql


# ── temporal_trend ────────────────────────────────────────────────────────────

def test_temporal_trend_year_normalization():
    ir = _ir(
        intent="temporal_trend",
        select=[{"col": "*", "agg": "COUNT", "alias": "n"}],
        group_by=["year"],
        order_by=[{"col": "year", "dir": "ASC"}],
    )
    sql = compile_ir(ir, WL)
    assert "year(timestamp)" in sql
    assert "GROUP BY year(timestamp)" in sql


def test_temporal_trend_multi_year_filter():
    ir = _ir(
        intent="temporal_trend",
        select=[{"col": "year(timestamp)", "agg": None, "alias": "yr"},
                {"col": "*", "agg": "COUNT", "alias": "n"}],
        filters=[{"table": "events", "col": "year(timestamp)", "op": "IN",
                  "val": [1999, 2000, 2001, 2002]}],
        group_by=["year(timestamp)"],
        temporal=None,
    )
    sql = compile_ir(ir, WL)
    assert "IN (1999, 2000, 2001, 2002)" in sql
    assert "GROUP BY year(timestamp)" in sql


# ── group_topk ────────────────────────────────────────────────────────────────

def test_group_topk_limit_and_order():
    ir = _ir(
        intent="group_topk",
        select=[{"col": "object_id", "agg": None, "alias": "cust"},
                {"col": "*", "agg": "COUNT", "alias": "n"}],
        group_by=["object_id"],
        order_by=[{"col": "n", "dir": "DESC"}],
        limit=5,
    )
    sql = compile_ir(ir, WL)
    assert "GROUP BY object_id" in sql
    assert "ORDER BY n DESC" in sql
    assert "LIMIT 5" in sql


# ── distinct count ────────────────────────────────────────────────────────────

def test_distinct_count():
    ir = _ir(select=[{"col": "object_id", "agg": "COUNT", "alias": "n", "distinct": True}])
    sql = compile_ir(ir, WL)
    assert "COUNT(DISTINCT object_id)" in sql


# ── delay_analysis ────────────────────────────────────────────────────────────

def test_delay_analysis_billing_to_ar_uses_object_attributes():
    ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "AVG", "alias": "avg_days"}],
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
    )
    sql = compile_ir(ir, WL)
    assert "FKDAT" in sql
    assert "AUGDT" in sql
    assert "billing_doc" in sql
    assert "ar_item" in sql


def test_delay_analysis_supports_stddev_and_quantile_aggs():
    stddev_ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "STDDEV", "alias": "stddev_days"}],
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
    )
    stddev_sql = compile_ir(stddev_ir, WL)
    assert "STDDEV(date_diff('day', b.FKDAT, ar.AUGDT)) AS stddev_days" in stddev_sql

    p90_ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "QUANTILE_CONT", "alias": "p90_days", "quantile": 0.9}],
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
    )
    p90_sql = compile_ir(p90_ir, WL)
    assert "QUANTILE_CONT(date_diff('day', b.FKDAT, ar.AUGDT), 0.9) AS p90_days" in p90_sql


def test_delay_analysis_other_relation_uses_event_timestamps():
    ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "AVG", "alias": "avg_days"}],
        joins=[{"relation_type": "order_to_delivery", "from_alias": "o",
                "to_alias": "d", "join_col": "from_object_id"}],
    )
    sql = compile_ir(ir, WL)
    assert "timestamp" in sql.lower()
    assert "FKDAT" not in sql
    assert "JOIN e2 ON e2.object_id = SPLIT_PART(links.e2_id, '_', 1)" in sql


def test_delay_analysis_delivery_to_billing_bridges_delivery_header_events():
    ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "AVG", "alias": "avg_days"}],
        joins=[{"relation_type": "delivery_to_billing", "from_alias": "d",
                "to_alias": "b", "join_col": "from_object_id"}],
    )
    sql = compile_ir(ir, WL)
    assert "event_type = 'delivery_created'" in sql
    assert "event_type = 'billing_created'" in sql
    assert "JOIN e1 ON e1.object_id = SPLIT_PART(links.e1_id, '_', 1)" in sql
    assert "JOIN e2 ON e2.object_id = links.e2_id" in sql


# ── anomaly_filter ────────────────────────────────────────────────────────────

def test_anomaly_filter_threshold():
    ir = _ir(
        intent="anomaly_filter",
        select=[{"col": "*", "agg": "COUNT", "alias": "n"}],
        filters=[{"col": "delay_days", "op": ">", "val": 873}],
        joins=[{"relation_type": "billing_to_ar", "from_alias": "b",
                "to_alias": "ar", "join_col": "from_object_id"}],
    )
    sql = compile_ir(ir, WL)
    assert "> 873" in sql
    assert "FKDAT" in sql


# ── path_relation negation ────────────────────────────────────────────────────

def test_path_negation_left_join_is_null():
    ir = _ir(
        intent="path_relation",
        negation=True,
        tables=["objects"],
        select=[{"col": "object_id", "agg": "COUNT", "alias": "n", "distinct": True}],
        filters=[{"table": "objects", "col": "object_type", "op": "=", "val": "order_item"}],
        joins=[{"relation_type": "order_to_delivery", "from_alias": "ord",
                "to_alias": "del", "join_col": "from_object_id"}],
    )
    sql = compile_ir(ir, WL)
    assert "LEFT JOIN relations" in sql
    assert "IS NULL" in sql
    assert "order_to_delivery" in sql


def test_orders_with_dunning_customer_template():
    ir = _ir(
        intent="path_relation",
        semantic_template="orders_with_dunning_customer",
        tables=["objects", "relations", "events"],
        select=[
            {
                "col": "object_id",
                "agg": "COUNT",
                "alias": "n_orders_with_dunning_customer",
                "distinct": True,
            }
        ],
        filters=[
            {"table": "objects", "col": "object_type", "op": "=", "val": "order_item"},
            {"table": "events", "col": "event_type", "op": "=", "val": "dunning_raised"},
        ],
        joins=[
            {
                "relation_type": "order_to_customer",
                "from_alias": "order",
                "to_alias": "customer",
                "join_col": "from_object_id",
            }
        ],
    )
    sql = compile_ir(ir, WL)
    assert "EXISTS" in sql
    assert "JOIN events e ON e.object_id = r.to_object_id" in sql
    assert "r.relation_type = 'order_to_customer'" in sql
    assert "e.event_type = 'dunning_raised'" in sql
    assert "COUNT(DISTINCT o.object_id)" in sql


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (
            "order_items_with_delivery_and_billing",
            ["order_to_delivery", "order_to_billing", "COUNT(DISTINCT r1.from_object_id)"],
        ),
        (
            "billing_docs_without_payment_clearing",
            ["billing_to_ar", "payment_clearing", "NOT IN"],
        ),
        (
            "order_items_billed_not_delivered",
            ["order_to_billing", "order_to_delivery", "NOT IN"],
        ),
        (
            "order_items_only_order_created",
            ["COUNT(DISTINCT event_type) = 1", "MAX(event_type) = 'order_created'"],
        ),
        (
            "customers_more_than_average_order_items",
            ["WITH c AS", "relation_type = 'order_to_customer'", "AVG(n)"],
        ),
        (
            "customers_with_delivery_events_in_period",
            [
                "order_to_customer",
                "order_to_delivery",
                "delivery_created",
                "SPLIT_PART(od.to_object_id, '_', 1)",
                "year(e.timestamp) = 2010",
                "month(e.timestamp) = 5",
            ],
        ),
    ],
)
def test_business_semantic_templates(template, expected):
    ir = _ir(
        intent="conformance",
        semantic_template=template,
        tables=["objects", "relations", "events"],
        select=[{"col": "*", "agg": "COUNT", "alias": "n"}],
        joins=[
            {"relation_type": "order_to_delivery"},
            {"relation_type": "order_to_billing"},
            {"relation_type": "billing_to_ar"},
            {"relation_type": "order_to_customer"},
        ],
    )
    if template == "customers_with_delivery_events_in_period":
        ir["filters"] = [
            {"table": "events", "col": "year(timestamp)", "op": "=", "val": 2010},
            {"table": "events", "col": "month(timestamp)", "op": "=", "val": 5},
        ]
    sql = compile_ir(ir, WL)
    for fragment in expected:
        assert fragment in sql


# ── join whitelist enforcement ────────────────────────────────────────────────

def test_invalid_join_raises():
    ir = _ir(
        intent="path_relation",
        joins=[{"relation_type": "fake_join", "from_alias": "x",
                "to_alias": "y", "join_col": "from_object_id"}],
    )
    with pytest.raises(IRCompileError, match="not in whitelist"):
        compile_ir(ir, WL)


def test_delay_analysis_invalid_join_raises():
    ir = _ir(
        intent="delay_analysis",
        select=[{"col": "*", "agg": "AVG", "alias": "d"}],
        joins=[{"relation_type": "not_a_real_join", "from_alias": "a",
                "to_alias": "b", "join_col": "from_object_id"}],
    )
    with pytest.raises(IRCompileError):
        compile_ir(ir, WL)


# ── BETWEEN operator ──────────────────────────────────────────────────────────

def test_between_operator():
    ir = _ir(filters=[
        {"table": "events", "col": "year(timestamp)", "op": "BETWEEN", "val": [2019, 2021]}
    ])
    sql = compile_ir(ir, WL)
    assert "BETWEEN 2019 AND 2021" in sql
