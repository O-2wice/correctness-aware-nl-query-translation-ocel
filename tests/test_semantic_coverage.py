"""Tests for question-level semantic coverage checks."""

from nl2ocel.semantic_coverage import check_semantic_coverage


def test_rejects_dunning_question_when_sql_drops_dunning_event():
    question = "How many order items are linked to a customer that received a dunning notice?"
    ir = {
        "intent": "path_relation",
        "tables": ["events", "relations"],
        "select": [{"col": "object_id", "agg": "COUNT", "alias": "n_order_items", "distinct": True}],
        "filters": [{"table": "events", "col": "object_type", "op": "=", "val": "order_item"}],
        "joins": [{"relation_type": "order_to_customer"}],
    }
    sql = """
        SELECT COUNT(DISTINCT events.object_id) AS n_order_items
        FROM events
        JOIN relations AS r ON events.object_id = r.from_object_id
        WHERE events.object_type = 'order_item'
          AND r.relation_type = 'order_to_customer'
    """

    result = check_semantic_coverage(question, ir, sql)

    assert not result.ok
    assert any("dunning_raised" in error for error in result.errors)


def test_accepts_dunning_question_when_required_event_and_relation_are_present():
    question = "How many order items are linked to a customer that received a dunning notice?"
    ir = {
        "intent": "path_relation",
        "tables": ["objects", "relations", "events"],
        "select": [{"col": "object_id", "agg": "COUNT", "alias": "n", "distinct": True}],
        "filters": [
            {"table": "objects", "col": "object_type", "op": "=", "val": "order_item"},
            {"table": "events", "col": "event_type", "op": "=", "val": "dunning_raised"},
        ],
        "joins": [{"relation_type": "order_to_customer"}],
    }
    sql = """
        SELECT COUNT(DISTINCT o.object_id) AS n
        FROM objects o
        WHERE o.object_type = 'order_item'
          AND EXISTS (
            SELECT 1 FROM relations r
            JOIN events e ON e.object_id = r.to_object_id
            WHERE r.from_object_id = o.object_id
              AND r.relation_type = 'order_to_customer'
              AND e.event_type = 'dunning_raised'
          )
    """

    result = check_semantic_coverage(question, ir, sql)

    assert result.ok


def test_rejects_delivery_billing_order_item_question_when_delivery_relation_missing():
    question = "How many order items are linked to both delivery and billing documents?"
    ir = {
        "intent": "path_relation",
        "tables": ["relations"],
        "select": [{"col": "from_object_id", "agg": "COUNT", "alias": "n", "distinct": True}],
        "filters": [],
        "joins": [{"relation_type": "order_to_billing"}],
    }
    sql = "SELECT COUNT(DISTINCT from_object_id) AS n FROM relations WHERE relation_type = 'order_to_billing'"

    result = check_semantic_coverage(question, ir, sql)

    assert not result.ok
    assert any("order_to_delivery" in error for error in result.errors)


def test_accepts_billing_payment_delay_with_attribute_based_compiler_sql():
    question = "What is the average number of days from billing creation to payment clearing?"
    ir = {
        "intent": "delay_analysis",
        "tables": ["relations"],
        "select": [{"col": "*", "agg": "AVG", "alias": "avg_days"}],
        "filters": [],
        "joins": [{"relation_type": "billing_to_ar"}],
    }
    sql = """
        SELECT AVG(date_diff('day', b.FKDAT, ar.AUGDT)) AS avg_days
        FROM relations r
        JOIN objects b ON b.object_id = r.from_object_id AND b.object_type = 'billing_doc'
        JOIN objects ar ON ar.object_id = r.to_object_id AND ar.object_type = 'ar_item'
        WHERE r.relation_type = 'billing_to_ar'
    """

    result = check_semantic_coverage(question, ir, sql)

    assert result.ok


def test_guards_remaining_catalog_event_types():
    change_result = check_semantic_coverage(
        "How many change events are in the log?",
        {"intent": "count_filter", "tables": ["events"], "select": [], "filters": []},
        "SELECT COUNT(*) AS n FROM events",
    )
    status_result = check_semantic_coverage(
        "How many order status change events are in the log?",
        {"intent": "count_filter", "tables": ["events"], "select": [], "filters": []},
        "SELECT COUNT(*) AS n FROM events",
    )

    assert not change_result.ok
    assert any("change_event" in error for error in change_result.errors)
    assert not status_result.ok
    assert any("order_status_change" in error for error in status_result.errors)


def test_guards_explicit_multi_year_questions():
    result = check_semantic_coverage(
        "Show billing creation event counts in 1994, 1997, 2010 and 2011.",
        {
            "intent": "temporal_trend",
            "tables": ["events"],
            "select": [],
            "filters": [
                {"table": "events", "col": "event_type", "op": "=", "val": "billing_created"},
                {"table": "events", "col": "year(timestamp)", "op": "IN", "val": [1994, 1997]},
            ],
        },
        """
        SELECT year(timestamp) AS year, COUNT(*) AS n
        FROM events
        WHERE event_type = 'billing_created'
          AND year(timestamp) IN (1994, 1997)
        GROUP BY year(timestamp)
        """,
    )

    assert not result.ok
    assert any("2010" in error for error in result.errors)
    assert any("2011" in error for error in result.errors)


def test_guards_customer_delivery_event_object_relation_path():
    result = check_semantic_coverage(
        "How many customers had deliveries in May 2010?",
        {
            "intent": "count_filter",
            "tables": ["events"],
            "select": [],
            "filters": [
                {"table": "events", "col": "event_type", "op": "=", "val": "delivery_created"},
                {"table": "events", "col": "year(timestamp)", "op": "=", "val": 2010},
                {"table": "events", "col": "month(timestamp)", "op": "=", "val": 5},
            ],
        },
        """
        SELECT COUNT(DISTINCT object_id) AS n
        FROM events
        WHERE event_type = 'delivery_created'
          AND year(timestamp) = 2010
          AND month(timestamp) = 5
        """,
    )

    assert not result.ok
    assert any("order_to_customer" in error for error in result.errors)
    assert any("order_to_delivery" in error for error in result.errors)


def test_accepts_customer_delivery_event_object_relation_path():
    result = check_semantic_coverage(
        "How many customers had deliveries in May 2010?",
        {
            "intent": "path_relation",
            "tables": ["relations", "events"],
            "select": [],
            "filters": [
                {"table": "events", "col": "event_type", "op": "=", "val": "delivery_created"},
                {"table": "events", "col": "year(timestamp)", "op": "=", "val": 2010},
                {"table": "events", "col": "month(timestamp)", "op": "=", "val": 5},
            ],
            "joins": [
                {"relation_type": "order_to_customer"},
                {"relation_type": "order_to_delivery"},
            ],
        },
        """
        SELECT COUNT(DISTINCT oc.to_object_id) AS n
        FROM relations oc
        JOIN relations od
          ON od.from_object_id = oc.from_object_id
         AND od.relation_type = 'order_to_delivery'
        JOIN events e
          ON e.object_id = SPLIT_PART(od.to_object_id, '_', 1)
        WHERE oc.relation_type = 'order_to_customer'
          AND e.event_type = 'delivery_created'
          AND year(e.timestamp) = 2010
          AND month(e.timestamp) = 5
        """,
    )

    assert result.ok
