from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None


ID_COLUMNS = {
    "event_id",
    "object_id",
    "from_object_id",
    "to_object_id",
}

TYPE_COLUMNS = {
    "event_type",
    "object_type",
    "from_object_type",
    "to_object_type",
    "relation_type",
}

MEASURE_COLUMNS = {"NETWR", "WRBTR"}


def _resolve_project_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.resolve()
    return Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
        return str(value)
    return value


def _semantic_role(col: str, series: pd.Series) -> str | None:
    if col in ID_COLUMNS:
        return "identifier"
    if col in TYPE_COLUMNS:
        return "category"
    if col == "source_table":
        return "lineage"
    if col in MEASURE_COLUMNS:
        return "measure"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return None


def _column_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = len(df)
    out: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        non_null = s.dropna()
        record: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "nullable": null_count > 0,
            "null_count": null_count,
            "null_pct": round((null_count / rows) if rows else 0.0, 6),
            "distinct_non_null": int(non_null.nunique(dropna=True)),
        }
        role = _semantic_role(col, s)
        if role:
            record["semantic_role"] = role
        out.append(record)
    return out


def _table_profile(name: str, df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "name": name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": _column_profile(df),
    }

    # Lightweight key-candidate hint: unique non-null columns.
    key_candidates: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.isna().any():
            continue
        if int(s.nunique(dropna=False)) == len(df):
            key_candidates.append(col)
    if key_candidates:
        profile["unique_non_null_columns"] = key_candidates

    if "event_type" in df.columns:
        profile["event_type_distribution"] = (
            df["event_type"].value_counts(dropna=False).to_dict()
        )
    if "object_type" in df.columns:
        profile["object_type_distribution"] = (
            df["object_type"].value_counts(dropna=False).to_dict()
        )
    if "relation_type" in df.columns:
        profile["relation_type_distribution"] = (
            df["relation_type"].value_counts(dropna=False).to_dict()
        )
    return profile


def _time_policy(tables: dict[str, pd.DataFrame], reference_date: pd.Timestamp) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for table_name, df in tables.items():
        for col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            s = df[col].dropna()
            if s.empty:
                checks.append(
                    {
                        "table": table_name,
                        "column": col,
                        "min": None,
                        "max": None,
                        "future_rows": 0,
                    }
                )
                continue
            # Normalize timezone handling to avoid tz-aware vs tz-naive comparison failures.
            s_cmp = s
            if getattr(s.dt, "tz", None) is not None:
                s_cmp = s.dt.tz_convert(None)
            future_rows = int((s_cmp > reference_date).sum())
            checks.append(
                {
                    "table": table_name,
                    "column": col,
                    "min": s_cmp.min().isoformat(),
                    "max": s_cmp.max().isoformat(),
                    "future_rows": future_rows,
                }
            )

    total_future_rows = int(sum(c["future_rows"] for c in checks))
    return {
        "reference_date_utc": reference_date.date().isoformat(),
        "mode": "drop_at_query_time",
        "total_future_rows_detected": total_future_rows,
        "checks": checks,
        "query_guard_example": (
            f"WHERE timestamp <= DATE '{reference_date.date().isoformat()}'"
        ),
    }


def _relation_whitelist(rel_df: pd.DataFrame, reference_date: pd.Timestamp) -> dict[str, Any]:
    transitions = (
        rel_df.groupby(["relation_type", "from_object_type", "to_object_type"])
        .size()
        .reset_index(name="edge_count")
        .sort_values(["relation_type", "edge_count"], ascending=[True, False])
    )
    transitions_list = transitions.to_dict(orient="records")

    return {
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_only_from_relations_parquet": True,
        "reference_date_utc": reference_date.date().isoformat(),
        "allowed_relation_types": sorted(rel_df["relation_type"].dropna().unique().tolist()),
        "allowed_object_type_transitions": transitions_list,
        "allowed_sql_joins": [
            {
                "name": "events_to_objects",
                "left_table": "events",
                "right_table": "objects",
                "on": [["events.object_id", "objects.object_id"]],
                "purpose": "Attach object attributes to events.",
            },
            {
                "name": "relations_from_to_objects",
                "left_table": "relations",
                "right_table": "objects",
                "on": [["relations.from_object_id", "objects.object_id"]],
                "purpose": "Resolve source-side relation objects.",
            },
            {
                "name": "relations_to_to_objects",
                "left_table": "relations",
                "right_table": "objects",
                "on": [["relations.to_object_id", "objects.object_id"]],
                "purpose": "Resolve target-side relation objects.",
            },
            {
                "name": "events_to_relations_from",
                "left_table": "events",
                "right_table": "relations",
                "on": [["events.object_id", "relations.from_object_id"]],
                "purpose": "Traverse event object to relation source object.",
            },
            {
                "name": "events_to_relations_to",
                "left_table": "events",
                "right_table": "relations",
                "on": [["events.object_id", "relations.to_object_id"]],
                "purpose": "Traverse event object to relation target object.",
            },
        ],
        "blocked_defaults": {
            "cross_join": True,
            "join_without_equality_predicate": True,
            "relation_type_not_in_whitelist": True,
        },
    }


def _query_probes(ocel_dir: Path, relation_types: list[str]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    if duckdb is not None:
        engine = "duckdb"
        con: Any = duckdb.connect(database=":memory:")  # type: ignore
        e = str((ocel_dir / "events.parquet").as_posix()).replace("'", "''")
        o = str((ocel_dir / "objects.parquet").as_posix()).replace("'", "''")
        r = str((ocel_dir / "relations.parquet").as_posix()).replace("'", "''")
        con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{e}')")
        con.execute(f"CREATE VIEW objects AS SELECT * FROM read_parquet('{o}')")
        con.execute(f"CREATE VIEW relations AS SELECT * FROM read_parquet('{r}')")
        future_sql = "SELECT COUNT(*) AS n FROM events WHERE timestamp > CURRENT_DATE"
    else:
        engine = "sqlite"
        con = sqlite3.connect(":memory:")
        pd.read_parquet(ocel_dir / "events.parquet").to_sql(
            "events", con, index=False, if_exists="replace"
        )
        pd.read_parquet(ocel_dir / "objects.parquet").to_sql(
            "objects", con, index=False, if_exists="replace"
        )
        pd.read_parquet(ocel_dir / "relations.parquet").to_sql(
            "relations", con, index=False, if_exists="replace"
        )
        future_sql = "SELECT COUNT(*) AS n FROM events WHERE date(timestamp) > date('now')"

    sqls: list[tuple[str, str]] = [
        ("events_count", "SELECT COUNT(*) AS n FROM events"),
        ("objects_count", "SELECT COUNT(*) AS n FROM objects"),
        ("relations_count", "SELECT COUNT(*) AS n FROM relations"),
        ("event_types", "SELECT event_type, COUNT(*) AS n FROM events GROUP BY 1 ORDER BY 2 DESC"),
        ("object_types", "SELECT object_type, COUNT(*) AS n FROM objects GROUP BY 1 ORDER BY 2 DESC"),
        ("relation_types", "SELECT relation_type, COUNT(*) AS n FROM relations GROUP BY 1 ORDER BY 2 DESC"),
        ("events_with_object_match", "SELECT COUNT(*) AS n FROM events e JOIN objects o ON e.object_id = o.object_id"),
        (
            "relations_from_object_match",
            "SELECT COUNT(*) AS n FROM relations r JOIN objects o ON r.from_object_id = o.object_id",
        ),
        (
            "relations_to_object_match",
            "SELECT COUNT(*) AS n FROM relations r JOIN objects o ON r.to_object_id = o.object_id",
        ),
        (
            "events_via_relations_from",
            "SELECT COUNT(*) AS n FROM events e JOIN relations r ON e.object_id = r.from_object_id",
        ),
        (
            "events_via_relations_to",
            "SELECT COUNT(*) AS n FROM events e JOIN relations r ON e.object_id = r.to_object_id",
        ),
        ("events_min_ts", "SELECT MIN(timestamp) AS min_ts FROM events"),
        ("events_max_ts", "SELECT MAX(timestamp) AS max_ts FROM events"),
        ("future_events", future_sql),
        (
            "billing_to_ar_delay_probe",
            """
            WITH bill AS (
                SELECT object_id, timestamp AS bill_ts
                FROM events
                WHERE event_type = 'billing_created' AND object_type = 'billing_doc'
            ),
            pay AS (
                SELECT object_id, timestamp AS pay_ts
                FROM events
                WHERE event_type = 'payment_clearing' AND object_type = 'ar_item'
            ),
            links AS (
                SELECT from_object_id AS bill_id, to_object_id AS ar_id
                FROM relations
                WHERE relation_type = 'billing_to_ar'
            )
            SELECT COUNT(*) AS n_linked
            FROM links
            JOIN bill ON bill.object_id = links.bill_id
            LEFT JOIN pay ON pay.object_id = links.ar_id
            """,
        ),
    ]

    for relation_type in relation_types:
        safe_rt = relation_type.replace("'", "''")
        sqls.append(
            (
                f"relation_{relation_type}",
                f"SELECT COUNT(*) AS n FROM relations WHERE relation_type = '{safe_rt}'",
            )
        )

    for probe_id, sql in sqls:
        try:
            con.execute(sql).fetchall()
            probes.append({"id": probe_id, "status": "pass"})
        except Exception as exc:
            probes.append({"id": probe_id, "status": "fail", "error": str(exc)})

    passed = sum(1 for p in probes if p["status"] == "pass")
    failed = sum(1 for p in probes if p["status"] == "fail")
    return {
        "engine": engine,
        "duckdb_available": duckdb is not None,
        "passed": passed,
        "failed": failed,
        "total": len(probes),
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gate A schema catalog + relation whitelist from OCEL parquet artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to project root (defaults to script parent root).",
    )
    args = parser.parse_args()

    root = _resolve_project_root(args.project_root)
    ocel_dir = root / "data" / "processed" / "ocel"
    config_dir = root / "configs"
    report_dir = root / "outputs" / "reports"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "events": pd.read_parquet(ocel_dir / "events.parquet"),
        "objects": pd.read_parquet(ocel_dir / "objects.parquet"),
        "relations": pd.read_parquet(ocel_dir / "relations.parquet"),
    }
    reference_date = pd.Timestamp.utcnow()
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_convert(None)
    reference_date = reference_date.normalize()

    relation_types = sorted(tables["relations"]["relation_type"].dropna().unique().tolist())
    catalog = {
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "source_ocel_dir": str(ocel_dir),
        "table_names": list(tables.keys()),
        "tables": {name: _table_profile(name, df) for name, df in tables.items()},
        "event_types": sorted(tables["events"]["event_type"].dropna().unique().tolist()),
        "object_types": sorted(tables["objects"]["object_type"].dropna().unique().tolist()),
        "relation_types": relation_types,
        "time_policy": _time_policy(tables, reference_date),
        "join_policy": {
            "relation_whitelist_file": "configs/relation_whitelist.json",
            "derived_only_from_relations_parquet": True,
        },
    }
    relation_whitelist = _relation_whitelist(tables["relations"], reference_date)
    probes = _query_probes(ocel_dir, relation_types)

    schema_path = config_dir / "schema_catalog.json"
    whitelist_path = config_dir / "relation_whitelist.json"
    probe_report_path = report_dir / "schema_catalog_query_probes.json"

    schema_path.write_text(json.dumps(catalog, indent=2, default=_json_safe), encoding="utf-8")
    whitelist_path.write_text(
        json.dumps(relation_whitelist, indent=2, default=_json_safe), encoding="utf-8"
    )
    probe_report_path.write_text(json.dumps(probes, indent=2), encoding="utf-8")

    summary = {
        "schema_catalog": str(schema_path),
        "relation_whitelist": str(whitelist_path),
        "query_probes_report": str(probe_report_path),
        "query_probes_passed": probes["passed"],
        "query_probes_total": probes["total"],
        "future_rows_detected": catalog["time_policy"]["total_future_rows_detected"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
