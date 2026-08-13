"""
semantic_coverage.py

Question-to-query semantic coverage checks.

The typed IR verifier proves that a generated query is syntactically safe and
schema-valid. This module adds a lighter semantic check: if the natural
language question explicitly mentions an OCEL event, object, or relation
concept, the accepted IR/SQL must preserve that concept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable


_EVENT_OBJECT_TYPES = {
    "billing_created": "billing_doc",
    "change_event": "order_item",
    "delivery_created": "delivery_item",
    "dunning_raised": "customer",
    "goods_issue": "delivery_item",
    "order_created": "order_item",
    "order_status_change": "order_item",
    "payment_clearing": "ar_item",
    "pricing_condition": "order_item",
}

_RELATION_GRAPH = {
    "order_item": [
        ("customer", "order_to_customer"),
        ("delivery_item", "order_to_delivery"),
        ("billing_doc", "order_to_billing"),
        ("material", "order_to_material"),
        ("customer", "order_to_payer"),
    ],
    "delivery_item": [
        ("order_item", "order_to_delivery"),
        ("billing_doc", "delivery_to_billing"),
    ],
    "billing_doc": [
        ("order_item", "order_to_billing"),
        ("delivery_item", "delivery_to_billing"),
        ("ar_item", "billing_to_ar"),
    ],
    "customer": [
        ("order_item", "order_to_customer"),
        ("order_item", "order_to_payer"),
    ],
    "ar_item": [
        ("billing_doc", "billing_to_ar"),
    ],
    "material": [
        ("order_item", "order_to_material"),
    ],
}

_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


@dataclass
class SemanticCoverageResult:
    """Result of checking whether generated IR/SQL covers question semantics."""

    errors: list[str] = field(default_factory=list)
    requirements_checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalise(text: str) -> str:
    text = text.lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _has_any(text: str, phrases: Iterable[str]) -> bool:
    return any(_normalise(phrase) in text for phrase in phrases)


def _has_all(text: str, phrases: Iterable[str]) -> bool:
    return all(_normalise(phrase) in text for phrase in phrases)


def _query_text(ir: dict | None, sql: str | None) -> str:
    pieces = [sql or ""]
    if ir is not None:
        pieces.append(json.dumps(ir, sort_keys=True, default=str))
    return " ".join(pieces).lower()


def _present(haystack: str, tokens: Iterable[str]) -> bool:
    return any(token.lower() in haystack for token in tokens)


def _years_in_question(question: str) -> list[str]:
    """Extract explicit four-digit years from the user question."""
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    return list(dict.fromkeys(years))


def _months_in_question(question: str) -> list[int]:
    q = _normalise(question)
    months: list[int] = []
    for name, number in _MONTH_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", q) and number not in months:
            months.append(number)
    for match in re.findall(r"\bmonth\s+([1-9]|1[0-2])\b", q):
        number = int(match)
        if number not in months:
            months.append(number)
    return months


def _relation_path(start: str, end: str) -> list[str]:
    if start == end:
        return []

    queue: list[tuple[str, list[str]]] = [(start, [])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        for next_node, relation_type in _RELATION_GRAPH.get(node, []):
            if next_node in seen:
                continue
            next_path = path + [relation_type]
            if next_node == end:
                return next_path
            seen.add(next_node)
            queue.append((next_node, next_path))
    return []


def _require_any(
    result: SemanticCoverageResult,
    generated: str,
    label: str,
    tokens: Iterable[str],
) -> None:
    token_list = list(tokens)
    result.requirements_checked.append(label)
    if not _present(generated, token_list):
        expected = " or ".join(token_list)
        result.errors.append(
            f"Semantic coverage violation: question mentions {label}, "
            f"but generated query does not include {expected}."
        )


def check_semantic_coverage(
    question: str,
    ir: dict | None,
    sql: str | None,
) -> SemanticCoverageResult:
    """
    Check that generated IR/SQL preserves explicit business concepts in q.

    This is intentionally schema-driven rather than question-id driven. It does
    not try to prove full denotation equivalence; it catches dangerous omissions
    such as a query mentioning dunning but producing SQL without dunning_raised.
    """
    q = _normalise(question)
    generated = _query_text(ir, sql)
    result = SemanticCoverageResult()

    # Event concepts. These fire only when the wording clearly asks about the
    # event type or uses a strong domain synonym.
    mentioned_events: list[str] = []
    event_requirements = [
        (
            "billing_created",
            (
                "billing created", "billing creation", "invoice creation",
                "invoice created", "invoice events", "billing events",
                "invoicing events",
            ),
            ("billing_created",),
        ),
        (
            "payment_clearing",
            (
                "payment clearing event", "payment clearing events",
                "payment cleared event", "clearing events",
                "cash clearing event", "cash clearing events",
            ),
            ("payment_clearing",),
        ),
        (
            "dunning_raised",
            ("dunning", "duning", "dunned", "overdue notice"),
            ("dunning_raised",),
        ),
        (
            "delivery_created",
            (
                "delivery created", "delivery creation", "delivery event",
                "delivery events", "deliveries", "delivered",
                "shipment event", "shipment events",
            ),
            ("delivery_created",),
        ),
        (
            "goods_issue",
            ("goods issue", "goods issued", "issue goods"),
            ("goods_issue",),
        ),
        (
            "pricing_condition",
            ("pricing condition", "price condition"),
            ("pricing_condition",),
        ),
        (
            "order_created",
            ("order creation event", "order created event", "order creation"),
            ("order_created",),
        ),
        (
            "change_event",
            ("change event", "change events", "changed event", "changed events", "modification event"),
            ("change_event",),
        ),
        (
            "order_status_change",
            (
                "order status change", "order status changes",
                "sales order status change", "status change event",
            ),
            ("order_status_change",),
        ),
    ]
    is_billing_payment_delay = (
        _has_any(q, ("billing", "invoice"))
        and _has_any(q, ("payment", "clearing", "cash clearing", "cleared"))
        and _has_any(q, ("day", "days", "delay", "average", "time", "long", "percentile"))
    )
    for label, cues, tokens in event_requirements:
        if _has_any(q, cues):
            mentioned_events.append(label)
            if label == "billing_created" and is_billing_payment_delay:
                tokens = ("billing_created", "billing_to_ar", "fkdat")
            _require_any(result, generated, label, tokens)

    # Object concepts. A relation type is also acceptable when it carries the
    # object type implicitly, e.g. order_to_customer covers customer/order_item.
    mentioned_objects: list[str] = []
    object_requirements = [
        (
            "order_item",
            ("order item", "order items"),
            ("order_item", "order_to_delivery", "order_to_billing", "order_to_customer", "order_to_material", "order_to_payer"),
        ),
        (
            "billing_doc",
            ("billing document", "billing documents", "invoice document", "invoice documents"),
            ("billing_doc", "order_to_billing", "delivery_to_billing", "billing_to_ar"),
        ),
        (
            "delivery_item",
            ("delivery item", "delivery items"),
            ("delivery_item", "order_to_delivery", "delivery_to_billing"),
        ),
        (
            "customer",
            ("customer", "customers"),
            ("customer", "order_to_customer", "order_to_payer"),
        ),
        (
            "ar_item",
            ("ar item", "ar items", "accounts receivable item", "accounts receivable items"),
            ("ar_item", "billing_to_ar"),
        ),
        (
            "material",
            ("material", "materials"),
            ("material", "order_to_material"),
        ),
    ]
    for label, cues, tokens in object_requirements:
        if _has_any(q, cues):
            mentioned_objects.append(label)
            _require_any(result, generated, label, tokens)

    # E/O/R consistency: when a question mentions an object type and an event
    # type that lives on a different object type, require the relation path
    # that connects those object types in the OCEL graph.
    for object_type in mentioned_objects:
        for event_type in mentioned_events:
            event_object_type = _EVENT_OBJECT_TYPES.get(event_type)
            if not event_object_type or event_object_type == object_type:
                continue
            for relation_type in _relation_path(object_type, event_object_type):
                _require_any(
                    result,
                    generated,
                    f"{object_type} to {event_type} E/O/R path",
                    (relation_type,),
                )

    # Relation/path semantics. These are the checks that prevent broad counts
    # from being accepted when a path concept in the question was skipped.
    if _has_any(q, ("order item", "order items")) and _has_any(q, ("customer", "customers", "payer")):
        if _has_any(q, ("payer", "paying customer")):
            _require_any(result, generated, "order item to payer relation", ("order_to_payer",))
        else:
            _require_any(result, generated, "order item to customer relation", ("order_to_customer",))

    if _has_any(q, ("order item", "order items")) and _has_any(q, ("delivery", "delivered", "shipment", "shipped")):
        _require_any(result, generated, "order item to delivery relation", ("order_to_delivery",))

    if _has_any(q, ("order item", "order items")) and _has_any(q, ("billing", "billed", "invoice")):
        _require_any(result, generated, "order item to billing relation", ("order_to_billing",))

    if _has_all(q, ("delivery", "billing")) and not _has_any(q, ("order item", "order items")):
        _require_any(result, generated, "delivery to billing relation", ("delivery_to_billing", "order_to_billing"))

    if _has_any(q, ("billing", "invoice")) and _has_any(q, ("payment", "clearing", "cash clearing", "ar item", "cleared")):
        _require_any(
            result,
            generated,
            "billing to payment clearing evidence",
            ("billing_to_ar", "payment_clearing", "augdt", "cleared"),
        )

    if _has_any(q, ("order item", "order items")) and _has_any(q, ("material", "materials")):
        _require_any(result, generated, "order item to material relation", ("order_to_material",))

    for year in _years_in_question(question):
        result.requirements_checked.append(f"year {year}")
        if year not in generated:
            result.errors.append(
                f"Semantic coverage violation: question mentions year {year}, "
                "but generated query does not include that year."
            )

    for month in _months_in_question(question):
        result.requirements_checked.append(f"month {month}")
        if "month" not in generated or str(month) not in generated:
            result.errors.append(
                f"Semantic coverage violation: question mentions month {month}, "
                "but generated query does not include that month filter."
            )

    return result
