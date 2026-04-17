"""
schema_retriever.py  —  Gate D3

Schema Retriever: R(q, S, k) → S_k

Given a natural-language question q, returns the top-k most relevant
schema elements from the full schema catalog S.

Strategy: TF-IDF over schema term descriptions.
- Each schema element (table, column, relation) becomes a document.
- Query q is matched against all documents.
- Top-k by cosine similarity are returned as a compact schema slice.

No external dependencies beyond scikit-learn (already in requirements.txt).
Falls back to full schema if sklearn unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ── Business synonym expansion for event types ────────────────────────────────
# TF-IDF only matches literal tokens. Analysts say "invoice" not "billing_created".
# Adding synonyms here closes the lexical gap without touching the LLM prompt.
_EVENT_TYPE_SYNONYMS: dict[str, str] = {
    "billing_created":          "invoice invoicing billed bill document",
    "payment_clearing":         "payment cleared clearing settled settlement ar",
    "order_created":            "sales order purchase ordered",
    "delivery_created":         "shipment shipped dispatch delivered",
    "goods_issue":              "goods issued goods issue shipment outbound delivery",
    "dunning_raised":           "dunning notice reminder overdue late payment",
    "pricing_condition":        "price pricing condition discount surcharge",
    "change_event":             "change changed update modification",
    "order_status_change":      "order status change status update",
}


# ── Schema document builder ────────────────────────────────────────────────────

def _build_schema_documents(
    catalog: dict, whitelist: list[dict]
) -> list[dict]:
    """
    Convert schema_catalog + whitelist into a flat list of searchable documents.
    Each doc has: id, text, type, payload.
    """
    docs: list[dict] = []

    for tbl in catalog.get("tables", []):
        tname = tbl["table_name"]
        row_count = tbl.get("row_count", 0)

        # Table-level doc
        docs.append({
            "id":      f"table:{tname}",
            "type":    "table",
            "text":    f"{tname} table rows {row_count} {tname.replace('_', ' ')}",
            "payload": {"table": tname, "row_count": row_count},
        })

        # Column-level docs
        for col in tbl.get("columns", []):
            cname  = col["name"]
            dtype  = col.get("dtype", "")
            unique = col.get("n_unique", 0)
            samples = " ".join(str(v) for v in col.get("sample_values", [])[:3])
            docs.append({
                "id":      f"col:{tname}.{cname}",
                "type":    "column",
                "text":    (
                    f"{tname} {cname} {dtype} {cname.replace('_', ' ')} "
                    f"{samples} unique {unique}"
                ),
                "payload": {"table": tname, "col": cname, "dtype": dtype},
            })

    # Event-type docs (from events.event_type sample values)
    event_types = list(catalog.get("event_types", []))
    for tbl in catalog.get("tables", []):
        if tbl["table_name"] == "events":
            for col in tbl.get("columns", []):
                if col["name"] == "event_type":
                    for val in col.get("sample_values", []):
                        if val not in event_types:
                            event_types.append(val)
    for et in event_types:
        synonyms = _EVENT_TYPE_SYNONYMS.get(et, "")
        docs.append({
            "id":      f"event_type:{et}",
            "type":    "event_type",
            "text":    f"event type {et} {et.replace('_', ' ')} {synonyms}".strip(),
            "payload": {"event_type": et},
        })

    # Relation docs
    for rel in whitelist:
        rt   = rel["relation_type"]
        frm  = rel.get("from_object_type", "")
        to   = rel.get("to_object_type", "")
        n    = rel.get("n_links", 0)
        docs.append({
            "id":      f"relation:{rt}",
            "type":    "relation",
            "text":    (
                f"relation {rt} {rt.replace('_', ' ')} "
                f"from {frm} {frm.replace('_', ' ')} "
                f"to {to} {to.replace('_', ' ')} links {n}"
            ),
            "payload": {"relation_type": rt, "from": frm, "to": to},
        })

    return docs


# ── TF-IDF retriever ──────────────────────────────────────────────────────────

class SchemaRetriever:
    """
    TF-IDF schema retriever. Matches NL question to top-k schema elements.

    Usage:
        retriever = SchemaRetriever.from_configs(catalog_path, whitelist_path)
        schema_slice = retriever.retrieve(question, k=10)
    """

    def __init__(self, docs: list[dict]):
        self._docs = docs
        self._vectorizer = None
        self._matrix = None
        self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            texts = [d["text"] for d in self._docs]
            self._matrix = self._vectorizer.fit_transform(texts)
        except ImportError:
            self._vectorizer = None
            self._matrix = None

    def retrieve(self, question: str, k: int = 12) -> dict:
        """
        R(q, S, k) → compact schema slice dict.

        Returns a dict with top-k schema elements grouped by type,
        suitable for injecting into an LLM prompt.
        """
        if self._vectorizer is None or self._matrix is None:
            return self._full_schema_fallback()

        import numpy as np  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        q_vec  = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]

        selected = [self._docs[i] for i in top_idx if scores[i] > 0.0]
        if not selected:
            return self._full_schema_fallback()

        return self._format_slice(selected)

    def _format_slice(self, docs: list[dict]) -> dict:
        tables:    list[str]  = []
        columns:   list[dict] = []
        relations: list[dict] = []
        event_types: list[str] = []

        seen_tables: set = set()
        seen_cols:   set = set()
        seen_rels:   set = set()
        seen_et:     set = set()

        for doc in docs:
            t = doc["type"]
            p = doc["payload"]

            if t == "table" and p["table"] not in seen_tables:
                tables.append(p["table"])
                seen_tables.add(p["table"])

            elif t == "column":
                key = (p["table"], p["col"])
                if key not in seen_cols:
                    columns.append(p)
                    seen_cols.add(key)
                    if p["table"] not in seen_tables:
                        tables.append(p["table"])
                        seen_tables.add(p["table"])

            elif t == "relation":
                if p["relation_type"] not in seen_rels:
                    relations.append(p)
                    seen_rels.add(p["relation_type"])

            elif t == "event_type":
                if p["event_type"] not in seen_et:
                    event_types.append(p["event_type"])
                    seen_et.add(p["event_type"])

        # Guarantee: if any event_type matched, the events table must be in context
        # so the LLM knows the schema it's querying against.
        if event_types and "events" not in seen_tables:
            tables.insert(0, "events")

        return {
            "tables":      tables,
            "columns":     columns,
            "relations":   relations,
            "event_types": event_types,
        }

    def _full_schema_fallback(self) -> dict:
        return self._format_slice(self._docs)

    def to_prompt_text(self, question: str, k: int = 12) -> str:
        """Return a compact schema slice as a prompt-ready string."""
        sl = self.retrieve(question, k=k)
        lines: list[str] = ["## Relevant Schema"]

        if sl["tables"]:
            lines.append(f"Tables: {', '.join(sl['tables'])}")

        if sl["columns"]:
            by_table: dict[str, list] = {}
            for c in sl["columns"]:
                by_table.setdefault(c["table"], []).append(f"{c['col']} ({c['dtype']})")
            for tname, cols in by_table.items():
                lines.append(f"  {tname}: {', '.join(cols)}")

        if sl["event_types"]:
            lines.append(f"Event types: {', '.join(sl['event_types'])}")

        if sl["relations"]:
            lines.append("Relations (allowed joins):")
            for r in sl["relations"]:
                lines.append(f"  {r['relation_type']}: {r['from']} -> {r['to']}")

        return "\n".join(lines)

    @classmethod
    def from_configs(
        cls,
        catalog_path: str | Path,
        whitelist_path: str | Path,
    ) -> "SchemaRetriever":
        catalog   = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        whitelist = json.loads(Path(whitelist_path).read_text(encoding="utf-8"))["whitelist"]
        docs = _build_schema_documents(catalog, whitelist)
        return cls(docs)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    retriever = SchemaRetriever.from_configs(
        root / "configs" / "schema_catalog.json",
        root / "configs" / "relation_whitelist.json",
    )

    questions = [
        "How many billing documents were created in 2019?",
        "Which customers have the most order items?",
        "What is the average payment delay from billing to AR clearing?",
        "Show me the order-to-delivery path for late deliveries",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(retriever.to_prompt_text(q, k=10))
        print("-" * 60)
