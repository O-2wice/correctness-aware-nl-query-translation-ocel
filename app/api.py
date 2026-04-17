"""
api.py — Flask REST wrapper around the correctness-aware NL-to-SQL pipeline.

Exposes the pipeline as HTTP endpoints so external frontends (SAP Build Apps,
any web client) can query the SAP O2C OCEL without a Python environment.

Usage:
    python app/api.py
    # or production:
    flask --app app.api run --host 0.0.0.0 --port 8000

Endpoints:
    POST /query      — run a natural-language question through the pipeline
    GET  /query      — run a question from ?question=... for no-code clients
    GET  /query/<q>  — run a URL-encoded question as a record id/path value
    GET  /health     — liveness check
    GET  /schema     — return schema catalog and relation whitelist
    GET  /examples   — return example questions grouped by query class
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from flask import Flask, jsonify, request
from flask_cors import CORS

from nl2ocel.llm_client import api_key_for_backend, default_model_for_backend
from nl2ocel.pipeline import ConstrainedPipeline
from examples import EXAMPLES

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow SAP Build Apps (cross-origin browser requests)

# ── Pipeline singleton ────────────────────────────────────────────────────────

_pipeline: ConstrainedPipeline | None = None
DEFAULT_DEMO_QUESTION = "What is the total number of billing creation events?"

def _get_pipeline() -> ConstrainedPipeline:
    global _pipeline
    if _pipeline is None:
        backend = os.environ.get("NL2OCEL_BACKEND", "deepseek").strip().lower()
        model = os.environ.get("NL2OCEL_MODEL") or default_model_for_backend(backend)
        credential = api_key_for_backend(backend, os.environ.get("NL2OCEL_API_KEY"))
        _pipeline = ConstrainedPipeline.from_project_root(
            ROOT,
            backend=backend,
            model=model,
            api_key=credential,
        )
    return _pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _explain_failure(status: str, errors: list[str], exec_error: str | None) -> str:
    if status == "accept":
        return ""
    e = " ".join(errors).lower()
    if status == "reject":
        if "semantic coverage violation" in e:
            return (
                "The generated query was schema-valid but did not preserve every "
                "business concept in the question, so it was rejected instead of "
                "showing a misleading answer."
            )
        if "event_type" in e and ("not in" in e or "invalid" in e):
            return "The query used an event type that does not exist in this dataset."
        if "join" in e and ("whitelist" in e or "not allowed" in e):
            return "The query requires a join between object types that is not permitted."
        if "column" in e and ("not found" in e or "does not exist" in e):
            return "The query referenced a column that does not exist in the schema."
        return "The verifier found schema violations it could not repair automatically."
    if status == "exec_error":
        ee = (exec_error or "").lower()
        if "binder error" in ee or "referenced column" in ee:
            return "DuckDB could not resolve a column in the generated SQL."
        if "syntax error" in ee:
            return "The generated SQL contained a syntax error."
        return "The SQL query failed to execute."
    if status == "llm_error":
        return "The language model call failed. Check your API key and network."
    if status == "parse_error":
        return "The language model returned a response that could not be parsed as a valid query plan."
    if status == "repair":
        return "The verifier could not repair the query plan within the allowed number of attempts."
    return ""


def _request_question() -> str:
    raw_body = request.get_data(as_text=True).strip()
    body = request.get_json(silent=True)
    if body is None:
        body = request.get_json(force=True, silent=True)

    if body is None and raw_body:
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = raw_body

    if body is None and request.form:
        if "question" in request.form:
            return request.form.get("question", "").strip()
        if "id" in request.form:
            return request.form.get("id", "").strip()

        # Some clients send JSON text as a form key when no content type is set.
        first_key = next(iter(request.form.keys()), "")
        if first_key:
            try:
                body = json.loads(first_key)
            except json.JSONDecodeError:
                body = first_key

    if isinstance(body, str):
        return body.strip()

    if isinstance(body, dict):
        value = body.get("question") or body.get("id")
        if value is None and isinstance(body.get("record"), dict):
            value = body["record"].get("question") or body["record"].get("id")
        return str(value or "").strip()

    return ""


def _as_text_list(value) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_answer_text(columns: list[str], rows: list[list], max_rows: int = 10) -> str:
    if not rows:
        return "No rows returned."

    if len(rows) == 1 and len(rows[0]) == 1:
        label = columns[0] if columns else "value"
        return f"{label}: {rows[0][0]}"

    display_columns = columns or [f"column_{idx + 1}" for idx in range(len(rows[0]))]
    lines = [" | ".join(display_columns)]
    for row in rows[:max_rows]:
        lines.append(" | ".join(str(value) for value in row))

    if len(rows) > max_rows:
        lines.append(f"... showing first {max_rows} of {len(rows)} rows")

    return "\n".join(lines)


def _as_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _chart_payload(question: str, columns: list[str], rows: list[list], max_points: int = 50) -> dict:
    """Return a simple chart contract for SAP/no-code frontends."""
    if len(columns) < 2 or not rows:
        return {
            "chart_available": False,
            "chart_type": "",
            "chart_x": "",
            "chart_y": "",
            "chart_labels": [],
            "chart_values": [],
            "chart_text": "",
            "chart_points": [],
        }

    points = []
    labels: list[str] = []
    values: list[int | float] = []
    for row in rows[:max_points]:
        if len(row) < 2:
            continue
        value = _as_number(row[1])
        if value is None:
            continue
        label = str(row[0])
        labels.append(label)
        values.append(value)
        points.append({"label": label, "value": value})

    if not points:
        return {
            "chart_available": False,
            "chart_type": "",
            "chart_x": "",
            "chart_y": "",
            "chart_labels": [],
            "chart_values": [],
            "chart_text": "",
            "chart_points": [],
        }

    q = question.lower()
    x_name = columns[0]
    x_lower = x_name.lower()
    trend_words = ("trend", "year", "month", "running", "change", "over time", "per month", "per year")
    chart_type = "line" if any(word in q for word in trend_words) or x_lower in {"year", "month", "date"} else "bar"
    chart_text = "\n".join(f"{p['label']}: {p['value']}" for p in points)

    return {
        "chart_available": True,
        "chart_type": chart_type,
        "chart_x": x_name,
        "chart_y": columns[1],
        "chart_labels": labels,
        "chart_values": values,
        "chart_text": chart_text,
        "chart_points": points,
    }


def _run_query(question: str):
    question = question.strip() or DEFAULT_DEMO_QUESTION

    try:
        pipeline = _get_pipeline()
        result = pipeline.run(question)
    except Exception as exc:
        return jsonify({"error": f"Pipeline error: {exc}"}), 500

    # Serialise DataFrame → columns + rows
    columns: list[str] = []
    rows: list[list] = []
    if result.result_df is not None:
        columns = list(result.result_df.columns)
        rows = [
            [str(v) if not isinstance(v, (int, float, bool, type(None))) else v
             for v in row]
            for row in result.result_df.values.tolist()
        ]

    provenance = result.provenance or {}
    exec_error = result.exec_error or ""
    error_explanation = _explain_failure(result.status, result.verify_errors, result.exec_error)
    answer_text = _format_answer_text(columns, rows)
    chart = _chart_payload(result.question, columns, rows)

    return jsonify({
        "answer_text":       answer_text,
        "question":          result.question,
        "status":            result.status,
        "sql":               result.pred_sql,
        "intent":            result.ir.get("intent") if result.ir else "",
        "columns":           columns,
        "columns_text":      _as_text_list(columns),
        "rows":              rows,
        "rows_text":         json.dumps(rows[:10], default=str),
        "chart_available":   chart["chart_available"],
        "chart_type":        chart["chart_type"],
        "chart_x":           chart["chart_x"],
        "chart_y":           chart["chart_y"],
        "chart_labels":      chart["chart_labels"],
        "chart_values":      chart["chart_values"],
        "chart_text":        chart["chart_text"],
        "chart_points":      chart["chart_points"],
        "n_rows":            len(rows),
        "latency_s":         result.latency_s,
        "ir_attempts":       result.ir_attempts,
        "verify_errors":     result.verify_errors,
        "verify_errors_text": _as_text_list(result.verify_errors),
        "exec_error":        exec_error,
        "exec_error_text":   exec_error,
        "provenance":        provenance,
        "tables_used_text":  _as_text_list(provenance.get("tables_used")),
        "joins_used_text":   _as_text_list(provenance.get("joins_used")),
        "error_explanation": error_explanation,
        "error_explanation_text": error_explanation,
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return jsonify({
        "service": "O2C Insight Assistant API",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "examples": "GET /examples",
            "schema": "GET /schema",
            "query": "POST /query or GET /query?question=...",
        },
        "sample_query": {
            "method": "POST",
            "path": "/query",
            "json": {"question": "What is the total number of billing creation events?"},
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "pipeline": "correctness-aware NL-to-SQL v1.0"})


@app.get("/examples")
def examples():
    return jsonify({"examples": EXAMPLES})


@app.get("/schema")
def schema():
    try:
        catalog = json.loads((ROOT / "configs" / "schema_catalog.json").read_text())
        whitelist = json.loads((ROOT / "configs" / "relation_whitelist.json").read_text())
        return jsonify({"schema_catalog": catalog, "relation_whitelist": whitelist})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/query")
def query():
    return _run_query(_request_question())


@app.get("/query")
@app.get("/query/")
def query_from_parameter():
    return _run_query(request.args.get("question", ""))


@app.get("/query/<path:question>")
def query_from_path(question: str):
    return _run_query(question)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting NL-to-SQL API on http://0.0.0.0:{port}")
    print("Docs: GET /health  GET /examples  GET /schema  POST /query")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
