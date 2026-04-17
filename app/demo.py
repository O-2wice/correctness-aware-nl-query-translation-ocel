"""
demo.py — Streamlit interactive demo for the correctness-aware NL query pipeline.

Pipeline stages shown per query:
  1. Schema retrieval  — TF-IDF top-k schema slice narrows LLM context
  2. NL → IR           — LLM produces a typed intermediate representation (JSON)
  3. Verifier          — checks enum values, column names, join whitelist; repairs if needed
  4. IR → SQL          — deterministic compiler; no arbitrary joins allowed
  5. Execution         — DuckDB runs SQL against OCEL parquet views; result returned

Usage:
    streamlit run app/demo.py

Environment (optional — key can also be entered in the sidebar):
    DEEPSEEK_API_KEY  — DeepSeek API key
    NL2OCEL_BACKEND   — default backend override (deepseek | openai | anthropic | ollama)

Error codes:
    accept      — pipeline completed successfully
    reject      — verifier rejected IR after max repair attempts
    exec_error  — SQL compiled but DuckDB raised an error
    llm_error   — LLM call failed (bad key, network, model unavailable)
    parse_error — LLM response could not be parsed as valid IR JSON
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import streamlit as st

# ── Project root and src import path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from nl2ocel.llm_client import default_model_for_backend

from examples import STREAMLIT_EXAMPLES

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NL→SQL | SAP O2C OCEL",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ─────────────────────────────────────────────
# history: list of dicts {question, status, sql, n_rows, latency_s, ir_attempts}
if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_deepseek_key(api_key: str) -> tuple[bool, str]:
    """
    Send a minimal request to DeepSeek to verify the API key before running
    the full pipeline.  Returns (ok: bool, message: str).
    """
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, "Key valid."
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "HTTP 401 — key rejected by DeepSeek. Check that the key is complete and active."
        if e.code == 402:
            return False, "HTTP 402 — DeepSeek account has insufficient credits."
        return False, f"HTTP {e.code} — {e.reason}"
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"


def status_label(status: str) -> str:
    """Human-readable explanation for each pipeline status code."""
    return {
        "accept":      "✅ Query translated and executed successfully.",
        "reject":      "⚠️ Verifier rejected the IR after all repair attempts. "
                       "The question may require a query class not yet supported.",
        "repair":      "⚠️ Verifier could not repair the query plan within the repair budget.",
        "exec_error":  "❌ SQL compiled but DuckDB raised an error during execution.",
        "llm_error":   "❌ LLM call failed. Check API key, network, or backend status.",
        "parse_error": "❌ LLM response could not be parsed as valid IR JSON.",
    }.get(status, f"Unknown status: {status}")


def valid_event_types_text() -> str:
    """Return current event types from schema_catalog.json for user messages."""
    try:
        catalog = json.loads((ROOT / "configs" / "schema_catalog.json").read_text(encoding="utf-8"))
        event_types = catalog.get("event_types", [])
        if event_types:
            return ", ".join(f"`{e}`" for e in event_types)
    except Exception:
        pass
    return (
        "`billing_created`, `change_event`, `delivery_created`, `dunning_raised`, "
        "`goods_issue`, `order_created`, `order_status_change`, `payment_clearing`, "
        "`pricing_condition`"
    )


def explain_failure(
    status: str,
    errors: list[str],
    exec_error: str | None,
) -> str:
    """
    Plain-English failure explanation shown below the status banner.
    Pattern-matches common verifier and DuckDB error messages so users
    don't have to read raw code blocks to understand what went wrong.
    """
    if status == "reject":
        if not errors:
            return (
                "The verifier rejected the intermediate representation after "
                "all repair attempts. No further information is available."
            )
        e = " ".join(errors).lower()
        if "event_type" in e and ("not in" in e or "invalid" in e):
            return (
                "The LLM used an event type that does not exist in this dataset. "
                f"Valid types: {valid_event_types_text()}. "
                "Try rephrasing with one of those exact terms."
            )
        if "join" in e and ("whitelist" in e or "not allowed" in e):
            return (
                "The query requires a join between object types that is not in "
                "the relation whitelist. This protects against hallucinated "
                "multi-hop queries. Try asking about a direct event→object "
                "relationship instead of a multi-step path."
            )
        if "column" in e and ("not found" in e or "does not exist" in e):
            return (
                "The LLM referenced a column that does not exist in the schema. "
                "The verifier blocked compilation to prevent a SQL runtime error. "
                "Check the Typed IR tab to see which column was flagged."
            )
        return (
            "The verifier found schema violations it could not repair within "
            "2 attempts. Try rephrasing the question more specifically — "
            "for example, name the event type or object type explicitly."
        )

    if status == "exec_error":
        e = (exec_error or "").lower()
        if "binder error" in e or "referenced column" in e:
            return (
                "DuckDB could not resolve a column referenced in the SQL. "
                "The compiled query uses a column name that does not exist "
                "in the parquet views. Check the Generated SQL tab."
            )
        if "parser error" in e or "syntax error" in e:
            return (
                "The compiled SQL has a syntax error. This usually happens "
                "when a filter value contains special characters or when the "
                "IR contains an unexpected expression. Check the Generated SQL tab."
            )
        if "conversion error" in e or "type mismatch" in e or "cannot cast" in e:
            return (
                "DuckDB encountered a type mismatch — for example, a string "
                "filter value being compared to a numeric column. "
                "Check the filter values in the Typed IR tab."
            )
        return (
            "DuckDB raised an error during execution. The SQL compiled "
            "correctly but failed at runtime. See the Generated SQL tab "
            "for the full query and error message."
        )

    if status == "llm_error":
        return (
            "The LLM API call failed. Verify that the API key is valid and "
            "the network is reachable. Use the 'Validate key' button in the "
            "sidebar to test your key before running a query."
        )

    if status == "parse_error":
        return (
            "The LLM returned a response that could not be parsed as valid "
            "IR JSON. This can happen when the model adds extra prose around "
            "the JSON output. Running the same query again may succeed."
        )

    return ""


@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline(backend: str, api_key: str | None, _root: str):
    """
    Load and cache the ConstrainedPipeline.

    Cached by (backend, api_key) so a new pipeline is created when the key
    changes.  _root is prefixed with underscore so Streamlit does not hash it
    (Path is not hashable by default).
    """
    from nl2ocel.pipeline import ConstrainedPipeline
    model = default_model_for_backend(backend)
    return ConstrainedPipeline.from_project_root(
        Path(_root),
        backend=backend,
        model=model,
        api_key=api_key,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    backend = st.selectbox(
        "LLM Backend",
        ["deepseek", "openai", "anthropic", "ollama"],
        help=(
            "deepseek → DeepSeek API (default, cheapest).  "
            "openai → OpenAI GPT-4o.  "
            "anthropic → Anthropic Claude Sonnet.  "
            "ollama → local qwen3:8b (no API spend)."
        ),
    )

    _env_key_map = {
        "deepseek":  "DEEPSEEK_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    _key_placeholder = {
        "deepseek":  "sk-…  (platform.deepseek.com)",
        "openai":    "sk-proj-…  (platform.openai.com)",
        "anthropic": "sk-ant-…  (console.anthropic.com)",
    }

    api_key: str | None = None
    if backend == "deepseek":
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            placeholder="sk-…",
            help="Get your key at platform.deepseek.com → API Keys.",
        )
        if api_key:
            if st.button("🔑 Validate key", use_container_width=True):
                with st.spinner("Testing key…"):
                    ok, msg = validate_deepseek_key(api_key)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.warning("Enter a DeepSeek API key to run queries.")
    elif backend in _env_key_map:
        env_var = _env_key_map[backend]
        api_key = st.text_input(
            f"{backend.capitalize()} API Key",
            type="password",
            value=os.environ.get(env_var, ""),
            placeholder=_key_placeholder[backend],
        )
        if not api_key:
            st.warning(f"Enter a {backend.capitalize()} API key to run queries.")
    else:
        st.info(
            "Ollama must be running locally on port 11434.  "
            "Start it with: `ollama serve`"
        )

    st.divider()

    # Example questions — clicking sets the text input via session state
    st.info(
        "**Core data range:** mostly 1994–2017  \n"
        "Known source outlier: `dunning_raised` has a documented future-year value, "
        "so demo questions use explicit years such as 2005/2006."
    )

    st.markdown("**Example questions**")
    for ex in STREAMLIT_EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.question_input = ex
            st.rerun()

    # Query history in sidebar
    if st.session_state.history:
        st.divider()
        st.markdown("**Recent queries**")
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            icon = "✅" if h["status"] == "accept" else "❌"
            with st.expander(f"{icon} {h['question'][:45]}…", expanded=False):
                st.caption(
                    f"Status: `{h['status']}` · "
                    f"{h['latency_s']:.1f}s · "
                    f"{h['ir_attempts']} attempt(s)"
                )
                if h["sql"]:
                    st.code(h["sql"], language="sql")
                if h["n_rows"] is not None:
                    st.caption(f"{h['n_rows']} row(s) returned")


# ── Main header ───────────────────────────────────────────────────────────────
st.title("🔍 Correctness-Aware NL Query Translation")
st.caption(
    "SAP O2C Event Log (OCEL 2.0) · Schema-constrained pipeline · DuckDB  |  "
    "[GitHub](https://github.com/O-2wice/correctness-aware-nl-query-translation-ocel)"
)

# Pipeline stage legend
with st.expander("ℹ️ How the pipeline works", expanded=False):
    st.markdown(
        """
| Stage | What happens |
|-------|-------------|
| **1 Schema retrieval** | TF-IDF selects the top-k most relevant tables/columns from `schema_catalog.json` to include in the LLM prompt |
| **2 NL → IR** | LLM converts the question into a typed Intermediate Representation (JSON) specifying intent, filters, joins, aggregation |
| **3 Verifier** | Checks enum values (e.g. valid event_type), column existence, join whitelist; triggers repair if invalid |
| **4 IR → SQL** | Deterministic compiler translates IR to DuckDB SQL; no arbitrary joins possible |
| **5 Execution** | DuckDB runs SQL against OCEL parquet views (events, objects, relations) |
        """
    )

st.divider()

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about the SAP O2C event log:",
    key="question_input",
    placeholder="e.g. How many billing_created events are there?",
)

run = st.button(
    "▶ Run Query",
    type="primary",
    disabled=not question or (backend == "deepseek" and not api_key),
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run and question:
    # Guard: validate DeepSeek key before burning a full pipeline call
    if backend == "deepseek":
        with st.spinner("Checking API key…"):
            ok, msg = validate_deepseek_key(api_key)
        if not ok:
            st.error(f"**API key rejected.** {msg}")
            st.stop()

    with st.spinner("Running pipeline (retrieval → IR → verify → compile → execute)…"):
        try:
            pipeline = load_pipeline(backend, api_key or None, str(ROOT))
            t0 = time.perf_counter()
            result = pipeline.run(question)
            wall_time = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"**Unexpected pipeline error:** {exc}")
            st.stop()

    # Save to history
    st.session_state.history.append({
        "question":    question,
        "status":      result.status,
        "sql":         result.pred_sql,
        "n_rows":      len(result.result_df) if result.result_df is not None else None,
        "latency_s":   result.latency_s,
        "ir_attempts": result.ir_attempts,
        "errors":      result.verify_errors or [],
        "exec_error":  result.exec_error,
    })

    # ── Status banner ─────────────────────────────────────────────────────────
    banner_fn = {
        "accept":      st.success,
        "reject":      st.warning,
        "exec_error":  st.error,
        "llm_error":   st.error,
        "parse_error": st.error,
    }.get(result.status, st.info)

    banner_fn(
        f"**{result.status.upper()}** · "
        f"Latency: {result.latency_s:.1f}s · "
        f"IR attempts: {result.ir_attempts}  \n"
        f"{status_label(result.status)}"
    )

    # Plain-English failure explanation (only shown on non-accept)
    if result.status != "accept":
        explanation = explain_failure(
            result.status,
            result.verify_errors or [],
            result.exec_error,
        )
        if explanation:
            st.info(f"**What went wrong:** {explanation}")

    # ── Pipeline stage detail tabs ────────────────────────────────────────────
    tab_ir, tab_sql, tab_result, tab_prov = st.tabs([
        "📐 Typed IR", "🗄️ Generated SQL", "📊 Result", "🔎 Provenance"
    ])

    with tab_ir:
        st.markdown("**Stage 2 output — Intermediate Representation**")
        st.caption(
            "The LLM converts your question into this structured JSON. "
            "The verifier checks it for schema validity before compilation."
        )
        if result.ir:
            st.json(result.ir)
            if result.verify_errors:
                st.warning("**Verifier errors (repair was attempted):**")
                for err in result.verify_errors:
                    st.code(err, language="text")
        else:
            st.warning("No IR produced.")
            if result.verify_errors:
                st.warning("**Verifier errors:**")
                for err in result.verify_errors:
                    st.code(err, language="text")

    with tab_sql:
        st.markdown("**Stage 4 output — Compiled SQL**")
        st.caption(
            "Deterministically compiled from the IR. "
            "Only joins present in `relation_whitelist.json` are permitted."
        )
        if result.pred_sql:
            st.code(result.pred_sql, language="sql")
            if result.exec_error:
                st.error(f"**DuckDB execution error:** {result.exec_error}")
        else:
            st.warning("No SQL generated.")

    with tab_result:
        st.markdown("**Stage 5 output — Query Result**")
        # Show query intent as context so results are interpretable
        if result.ir and result.ir.get("intent"):
            st.caption(f"Query class: `{result.ir['intent']}`")
        if result.result_df is not None and not result.result_df.empty:
            df = result.result_df.reset_index(drop=True)
            # Single-cell scalar result (e.g. COUNT) — show as a large metric
            if df.shape == (1, 1):
                col_name = df.columns[0]
                value    = df.iloc[0, 0]
                st.metric(label=col_name, value=f"{value:,}" if isinstance(value, (int, float)) else str(value))
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                # Warn when a numeric column is constant — often signals a bad GROUP BY
                for col in df.select_dtypes(include="number").columns:
                    if df[col].nunique() == 1:
                        st.warning(
                            f"Column `{col}` has the same value ({df[col].iloc[0]}) "
                            "in every row. The pipeline may have omitted the identifier "
                            "from SELECT or used the wrong aggregation. "
                            "Check the **Generated SQL** tab for details."
                        )
                        break
            st.caption(f"{len(df)} row(s) · {len(df.columns)} column(s)")
        elif result.status == "accept":
            st.info("Query executed successfully but returned 0 rows.")
        else:
            st.warning(
                result.exec_error
                or "No result — see IR and SQL tabs for details."
            )

    with tab_prov:
        st.markdown("**Pipeline provenance**")
        st.caption(
            "Captures intent, joins used, result fingerprint, grounding coverage, "
            "and the provenance query used for auditability."
        )
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Provenance**")
            st.json(result.provenance)
        with pcol2:
            st.markdown("**Schema slice retrieved (Stage 1)**")
            sl = result.schema_slice or {}
            # Show a readable breakdown instead of raw JSON
            if sl.get("event_types"):
                st.markdown(f"**Event types matched:** `{'`, `'.join(sl['event_types'])}`")
            if sl.get("tables"):
                st.markdown(f"**Tables in context:** `{'`, `'.join(sl['tables'])}`")
            if sl.get("relations"):
                st.markdown("**Relations in context:**")
                for r in sl["relations"]:
                    st.caption(
                        f"  `{r['relation_type']}` — "
                        f"{r.get('from','')} → {r.get('to','')}"
                    )
            if sl.get("columns"):
                st.markdown("**Columns matched:**")
                for c in sl["columns"]:
                    st.caption(f"  `{c.get('table','')}.{c.get('col','')}` — {c.get('dtype','')}")
            if not any(sl.get(k) for k in ("event_types","tables","relations","columns")):
                st.json(sl)

# ── Session history (main area) ───────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.markdown("### 📜 Session History")
    st.caption("All queries run this session. Expand a row to see the SQL.")

    import pandas as _pd

    _STATUS_ICON = {
        "accept": "✅", "reject": "⚠️",
        "exec_error": "❌", "llm_error": "❌", "parse_error": "❌",
    }

    for idx, h in enumerate(reversed(st.session_state.history)):
        icon   = _STATUS_ICON.get(h["status"], "•")
        rows   = f"{h['n_rows']} row(s)" if h["n_rows"] is not None else "—"
        label  = (
            f"{icon} **{h['question'][:70]}{'…' if len(h['question']) > 70 else ''}**  "
            f"`{h['status']}` · {h['latency_s']:.1f}s · {rows}"
        )
        with st.expander(label, expanded=False):
            if h["sql"]:
                st.code(h["sql"], language="sql")
            elif h["status"] != "accept":
                expl = explain_failure(h["status"], h.get("errors", []), h.get("exec_error"))
                if expl:
                    st.info(expl)
