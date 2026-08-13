"""Tests for deterministic semantic templates in the pipeline."""

from nl2ocel.pipeline import _semantic_template_ir


def test_detects_dunning_customer_order_item_question():
    ir = _semantic_template_ir(
        "How many order items are linked to a customer that received a dunning notice?"
    )
    assert ir is not None
    assert ir["semantic_template"] == "orders_with_dunning_customer"
    assert ir["joins"][0]["relation_type"] == "order_to_customer"
    assert any(f["val"] == "dunning_raised" for f in ir["filters"])


def test_detects_common_dunning_typo():
    ir = _semantic_template_ir(
        "How many order items are linked to a customer that received a duning notice?"
    )
    assert ir is not None
    assert ir["semantic_template"] == "orders_with_dunning_customer"


def test_unrelated_questions_do_not_use_template():
    assert _semantic_template_ir("How many payment clearing events are recorded?") is None


def test_detects_demo_business_templates():
    cases = {
        "How many order items are linked to both delivery and billing documents?":
            "order_items_with_delivery_and_billing",
        "How many billing documents have no corresponding payment clearing event?":
            "billing_docs_without_payment_clearing",
        "How many order items were billed but never delivered?":
            "order_items_billed_not_delivered",
        "How many order items have only an order creation event and no downstream activity?":
            "order_items_only_order_created",
        "How many customers have more than the average number of linked order items?":
            "customers_more_than_average_order_items",
        "How many customers had deliveries in May 2010?":
            "customers_with_delivery_events_in_period",
    }

    for question, template in cases.items():
        ir = _semantic_template_ir(question)
        assert ir is not None
        assert ir["semantic_template"] == template


def test_customer_delivery_period_template_keeps_period_and_path():
    ir = _semantic_template_ir("How many customers had deliveries in May 2010?")

    assert ir is not None
    assert {j["relation_type"] for j in ir["joins"]} == {
        "order_to_customer",
        "order_to_delivery",
    }
    assert any(f["col"] == "year(timestamp)" and f["val"] == 2010 for f in ir["filters"])
    assert any(f["col"] == "month(timestamp)" and f["val"] == 5 for f in ir["filters"])
