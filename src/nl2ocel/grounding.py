"""
grounding.py — source-row provenance layer for the constrained pipeline.

`pipeline.run()` records query metadata such as tables used, joins used, and
result row count. This module adds source-row traceback so each accepted answer
can be audited against the raw events, objects, and relations.

This module produces a `GroundingReport` per query containing:

  - `source_event_ids`   : up to MAX_SAMPLE event_ids that match the IR's
                           filter conjunction (drives count_filter, temporal_trend,
                           anomaly_filter)
  - `source_object_ids`  : up to MAX_SAMPLE object_ids that match the IR's
                           object-level filters (drives path_relation, conformance)
  - `source_relation_ids`: up to MAX_SAMPLE (from_object_id, to_object_id, relation_type)
                           tuples that contributed for path / delay / conformance
  - `coverage`           : full counts of source rows under each filter
  - `provenance_query`   : the actual SQL we executed to compute the above
                           (so the audit trail is itself reproducible)

The grounding query is derived from the IR — not from the compiled SQL —
because the IR carries a structured filter / join description we can rewrite
to a "give me the source IDs" query without re-parsing SQL. This keeps
grounding lightweight (one extra DuckDB scan per question, capped at
MAX_SAMPLE rows) and keeps the data flow inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Cap returned IDs so a single audit call never blows up memory. The full
# counts in `coverage` are unbounded; only the sample lists are capped.
MAX_SAMPLE = 100


@dataclass
class GroundingReport:
    """Per-query cell-level grounding evidence."""
    source_event_ids:    list[str] = field(default_factory=list)
    source_object_ids:   list[str] = field(default_factory=list)
    source_relation_ids: list[tuple] = field(default_factory=list)
    coverage:            dict[str, int] = field(default_factory=dict)
    provenance_query:    str = ""
    grounding_status:    str = "none"   # none | events | objects | relations | mixed
    notes:               str = ""

    def to_dict(self) -> dict:
        """Serialise for CSV export — list-valued fields are stringified."""
        return {
            "grounding_status":    self.grounding_status,
            "n_source_events":     self.coverage.get("events", 0),
            "n_source_objects":    self.coverage.get("objects", 0),
            "n_source_relations":  self.coverage.get("relations", 0),
            "sample_event_ids":    ";".join(self.source_event_ids[:MAX_SAMPLE]),
            "sample_object_ids":   ";".join(self.source_object_ids[:MAX_SAMPLE]),
            "sample_relation_keys": ";".join(
                f"{a}->{b}({c})" for a, b, c in self.source_relation_ids[:MAX_SAMPLE]
            ),
            "provenance_query":    self.provenance_query,
            "notes":               self.notes,
        }


def _escape(val: Any) -> str:
    """Single-quote SQL escape compatible with `_escape_str` in ir_to_sql."""
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    if val is None:
        return "NULL"
    return str(val)


def _build_event_filter(ir: dict) -> str:
    """Translate IR filters that target the events table into a SQL WHERE."""
    conds: list[str] = []
    for f in ir.get("filters", []):
        col = f.get("col", "")
        op  = f.get("op", "=").upper()
        val = f.get("val")
        # Skip filters that don't apply to events (object-only attributes).
        if col in {"object_type", "object_id", "event_type", "event_id"} or "(" in col \
           or col == "year(timestamp)" or col == "month(timestamp)" or col == "timestamp":
            if op == "IN" and isinstance(val, list):
                lst = ",".join(_escape(v) for v in val)
                conds.append(f"{col} IN ({lst})")
            elif op == "BETWEEN" and isinstance(val, list) and len(val) == 2:
                conds.append(f"{col} BETWEEN {_escape(val[0])} AND {_escape(val[1])}")
            else:
                conds.append(f"{col} {op} {_escape(val)}")
    # Temporal shorthand
    temporal = ir.get("temporal") or {}
    year = temporal.get("year")
    if year is not None:
        conds.append(f"year(timestamp) = {int(year)}")
    return " AND ".join(conds) if conds else "1=1"


def _build_object_filter(ir: dict) -> str:
    """WHERE on the objects table — only attribute / object_type filters."""
    conds: list[str] = []
    base_otype = (ir.get("base") or {}).get("object_type")
    if base_otype:
        conds.append(f"object_type = {_escape(base_otype)}")
    for f in ir.get("filters", []):
        col = f.get("col", "")
        op  = f.get("op", "=").upper()
        val = f.get("val")
        if col == "object_type" or col in {"NETWR", "WAERK", "ZTERM", "FKDAT", "AUGDT", "cleared"}:
            conds.append(f"{col} {op} {_escape(val)}")
    return " AND ".join(conds) if conds else "1=1"


def _build_relation_filter(ir: dict) -> str:
    """WHERE on the relations table — drives delay / path / conformance."""
    conds: list[str] = []
    joins = ir.get("joins") or []
    relation_types = [j.get("relation_type") for j in joins if j.get("relation_type")]
    # Conformance/IR may carry relation_type at top-level fields.
    for key in ("missing_relation_type", "has_relation_type", "also_missing_relation"):
        rt = ir.get(key)
        if rt:
            relation_types.append(rt)
    if relation_types:
        if len(relation_types) == 1:
            conds.append(f"relation_type = {_escape(relation_types[0])}")
        else:
            lst = ",".join(_escape(r) for r in relation_types)
            conds.append(f"relation_type IN ({lst})")
    return " AND ".join(conds) if conds else "1=1"


def compute_grounding(ir: dict, conn) -> GroundingReport:
    """Compute a `GroundingReport` for one IR by running 1-3 source-ID queries.

    The function is intent-aware: only the relevant tables are scanned.
    All scans use the same DuckDB connection as `pipeline.run()` so they
    benefit from the same cached parquet views.

    Parameters
    ----------
    ir : dict
        The verified IR that produced the answer. Must include `intent` and
        `tables`.
    conn : duckdb connection
        Open connection with `events`, `objects`, `relations` views already
        registered.

    Returns
    -------
    GroundingReport
    """
    intent = ir.get("intent", "")
    tables = set(ir.get("tables") or [])
    # Extended intents carry the table inside a `base` sub-dict.
    base = ir.get("base") or {}
    if base.get("table"):
        tables.add(base["table"])
    if base.get("object_type"):
        tables.add("objects")

    report = GroundingReport()
    queries: list[str] = []

    # ── Events grounding ────────────────────────────────────────────────────
    if "events" in tables and intent in {
        "count_filter", "temporal_trend", "group_topk",
        "anomaly_filter", "delay_analysis", "nested_agg", "window_agg",
    }:
        where = _build_event_filter(ir)
        q = f"SELECT event_id FROM events WHERE {where} LIMIT {MAX_SAMPLE}"
        queries.append(q)
        try:
            df = conn.execute(q).df()
            report.source_event_ids = df["event_id"].astype(str).tolist()
            full = conn.execute(
                f"SELECT COUNT(*) AS n FROM events WHERE {where}"
            ).fetchone()[0]
            report.coverage["events"] = int(full)
        except Exception as exc:
            report.notes += f"events grounding failed: {exc}; "

    # ── Objects grounding ──────────────────────────────────────────────────
    if "objects" in tables or intent in {"path_relation", "conformance"}:
        where = _build_object_filter(ir)
        q = f"SELECT object_id FROM objects WHERE {where} LIMIT {MAX_SAMPLE}"
        queries.append(q)
        try:
            df = conn.execute(q).df()
            report.source_object_ids = df["object_id"].astype(str).tolist()
            full = conn.execute(
                f"SELECT COUNT(*) AS n FROM objects WHERE {where}"
            ).fetchone()[0]
            report.coverage["objects"] = int(full)
        except Exception as exc:
            report.notes += f"objects grounding failed: {exc}; "

    # ── Relations grounding ────────────────────────────────────────────────
    if "relations" in tables or intent in {
        "path_relation", "delay_analysis", "anomaly_filter", "conformance",
    }:
        where = _build_relation_filter(ir)
        if where != "1=1":  # only emit if we actually have a relation_type filter
            q = (f"SELECT from_object_id, to_object_id, relation_type "
                 f"FROM relations WHERE {where} LIMIT {MAX_SAMPLE}")
            queries.append(q)
            try:
                df = conn.execute(q).df()
                report.source_relation_ids = list(zip(
                    df["from_object_id"].astype(str),
                    df["to_object_id"].astype(str),
                    df["relation_type"].astype(str),
                ))
                full = conn.execute(
                    f"SELECT COUNT(*) AS n FROM relations WHERE {where}"
                ).fetchone()[0]
                report.coverage["relations"] = int(full)
            except Exception as exc:
                report.notes += f"relations grounding failed: {exc}; "

    # ── Status summary ─────────────────────────────────────────────────────
    populated = [t for t in ("events", "objects", "relations") if report.coverage.get(t, 0) > 0]
    if not populated:
        report.grounding_status = "none"
    elif len(populated) == 1:
        report.grounding_status = populated[0]
    else:
        report.grounding_status = "mixed"

    report.provenance_query = "; ".join(queries)
    return report


if __name__ == "__main__":
    # Smoke test on a real OCEL.
    import duckdb
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    conn = duckdb.connect()
    for t in ("events", "objects", "relations"):
        conn.execute(
            f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{root / 'data/processed/ocel'}/{t}.parquet')"
        )
    sample_ir = {
        "intent":  "count_filter",
        "tables":  ["events"],
        "select":  [{"col": "*", "agg": "COUNT", "alias": "n"}],
        "filters": [
            {"table": "events", "col": "event_type", "op": "=", "val": "billing_created"},
            {"table": "events", "col": "year(timestamp)", "op": "=", "val": 2005},
        ],
    }
    rep = compute_grounding(sample_ir, conn)
    print("grounding status:", rep.grounding_status)
    print("coverage:", rep.coverage)
    print("first 3 event_ids:", rep.source_event_ids[:3])
    print("provenance_query:", rep.provenance_query[:200])
