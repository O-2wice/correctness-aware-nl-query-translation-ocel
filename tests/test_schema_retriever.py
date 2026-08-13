"""Unit tests for schema_retriever.py — TF-IDF retrieval + synonym expansion."""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG   = ROOT / "configs" / "schema_catalog.json"
WHITELIST = ROOT / "configs" / "relation_whitelist.json"


@pytest.fixture(scope="module")
def retriever():
    from nl2ocel.schema_retriever import SchemaRetriever
    return SchemaRetriever.from_configs(CATALOG, WHITELIST)


def test_retriever_loads(retriever):
    assert retriever is not None


def test_retrieve_returns_dict(retriever):
    sl = retriever.retrieve("How many billing documents were created?", k=10)
    assert isinstance(sl, dict)
    assert "tables" in sl
    assert "event_types" in sl
    assert "relations" in sl


def test_billing_question_retrieves_billing_created(retriever):
    sl = retriever.retrieve("How many invoices were created in 2019?", k=10)
    assert "billing_created" in sl["event_types"], \
        f"Expected billing_created, got: {sl['event_types']}"


def test_payment_question_retrieves_payment_clearing(retriever):
    sl = retriever.retrieve("What is the average payment delay after billing?", k=10)
    event_types = sl["event_types"]
    assert "payment_clearing" in event_types or "billing_created" in event_types


def test_retriever_indexes_all_catalog_event_types_not_only_samples(retriever):
    goods = retriever.retrieve("How many goods issue events are in the log?", k=10)
    pricing = retriever.retrieve("How many pricing condition events are in the log?", k=10)

    assert "goods_issue" in goods["event_types"]
    assert "pricing_condition" in pricing["event_types"]


def test_delivery_question_retrieves_delivery_relation(retriever):
    sl = retriever.retrieve("Which order items are linked to a delivery?", k=12)
    relation_types = [r["relation_type"] for r in sl["relations"]]
    assert "order_to_delivery" in relation_types


def test_billing_to_ar_retrieved_for_delay_question(retriever):
    sl = retriever.retrieve("What is the average delay from billing to AR clearing?", k=12)
    relation_types = [r["relation_type"] for r in sl["relations"]]
    assert "billing_to_ar" in relation_types


def test_events_table_included_when_event_type_matched(retriever):
    sl = retriever.retrieve("How many billing_created events?", k=5)
    assert "events" in sl["tables"]


def test_full_schema_fallback_returns_all_tables(retriever):
    sl = retriever._full_schema_fallback()
    assert "events" in sl["tables"]
    assert "objects" in sl["tables"]
    assert "relations" in sl["tables"]


def test_to_prompt_text_returns_string(retriever):
    text = retriever.to_prompt_text("How many order items?", k=8)
    assert isinstance(text, str)
    assert "## Relevant Schema" in text
    assert "Event types" in text or "Tables" in text
