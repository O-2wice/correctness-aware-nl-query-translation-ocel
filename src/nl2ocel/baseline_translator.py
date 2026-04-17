"""
baseline_translator.py  —  Gate C1

Three NL-to-SQL baseline methods for OCEL benchmark evaluation.
Canonical labels (match manuscript Chapter 3 and README):

B1  zero_shot   — Rajkumar et al. 2022, "Evaluating the Text-to-SQL
                  Capabilities of Large Language Models" (arXiv:2204.00498).
                  Create Table + Select 3 prompt: CREATE TABLE statements with
                  types, foreign-key comments, and 3 sample rows per table.

B2  few_shot    — Nan et al. 2023, "Enhancing Few-shot Text-to-SQL Capabilities
                  of Large Language Models: A Study on Prompt Design Strategies"
                  (Findings EMNLP 2023). Similarity-Diversity demonstration
                  selection via k-Means clustering of SQL syntax vectors,
                  code-sequence schema (CREATE TABLE) with semantic and
                  structure augmentation, and integrated voting over shot counts.

B3  din_sql     — Pourreza & Rafiei, 2023, "DIN-SQL: Decomposed In-Context
                  Learning of Text-to-SQL with Self-Correction" (NeurIPS 2023).
                  Stage 1: Schema Linking  (CoT + 10 few-shot demos)
                  Stage 2: Classification  (EASY / NON-NESTED / NESTED + demos)
                  Stage 3: SQL Generation  (class-specific prompt with demos;
                                            NatSQL IR for non-nested class)
                  Stage 4: Self-Correction (generic for CodeX-class models,
                                            gentle for GPT-4-class models)

Usage (standalone):
    python src/nl2ocel/baseline_translator.py \
        --mode din_sql \
        --benchmark benchmark/nl2ocel_benchmark_dev.csv \
        --catalog  configs/schema_catalog.json \
        --whitelist configs/relation_whitelist.json \
        --ocel_dir  data/processed/ocel \
        --out       outputs/reports/nb03_b3_results.csv

Usage (import):
    from nl2ocel.baseline_translator import BaselineTranslator
    t = BaselineTranslator(mode='din_sql', catalog=catalog, whitelist=whitelist)
    result = t.translate("How many billing events in 2005?")
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

try:
    from .result_hash import hash_dataframe
except ImportError:  # pragma: no cover - supports `python src/nl2ocel/...py`
    from result_hash import hash_dataframe

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_MODEL  = "qwen3:8b"
OLLAMA_URL     = "http://localhost:11434"
TEMPERATURE    = 0.0
MAX_TOKENS     = 2048
TIMEOUT_S      = 300

FEW_SHOT_QIDS = ["Q001", "Q019", "Q025"]   # easy / medium / hard; excluded from B3 eval


# ── LLM callers — delegates to shared llm_client (eliminates duplication) ──────

def call_ollama(prompt: str, model: str = DEFAULT_MODEL,
                timeout: int = 300, ollama_url: str = "http://localhost:11434",
                max_tokens: int | None = None, stop: list[str] | None = None) -> dict:
    from .llm_client import call_ollama as _call
    return _call(prompt, model=model, timeout=timeout, ollama_url=ollama_url,
                 max_tokens=max_tokens, stop=stop)


def call_deepseek(prompt: str, model: str = DEEPSEEK_MODEL,
                  api_key: str | None = None,
                  max_tokens: int | None = None, stop: list[str] | None = None) -> dict:
    from .llm_client import call_deepseek as _call
    return _call(prompt, model=model, api_key=api_key,
                 max_tokens=max_tokens, stop=stop)


# ── SQL extractor ─────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:^|\n)((?:SELECT|WITH|select|with)[\s\S]+?)(?:;\s*$|\n\n|$)", text)
    if m:
        return m.group(1).strip().rstrip(";")
    return text.strip()


# ── DIN-SQL stage-output extractors ──────────────────────────────────────────

def _extract_schema_links(raw_text: str) -> str:
    """Pull the `Schema_links: [...]` line out of a DIN-SQL stage-1 response.

    Falls back to the full CoT text if no canonical line is found so stage 2
    still has something to condition on. The paper relies on the model
    emitting this line verbatim at the end of the CoT (Figure 3a).
    """
    match = re.search(r"Schema_links\s*:\s*(\[[^\]]*\])", raw_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return raw_text.strip() or "[]"


def _extract_class_label(raw_text: str) -> str:
    """Pull the `Label: "EASY"/"NON-NESTED"/"NESTED"` token from stage 2.

    Defaults to "EASY" when no label is detected so stage 3 still routes to
    a valid template rather than crashing.
    """
    label_match = re.search(r'Label\s*:\s*"?(EASY|NON-NESTED|NESTED)"?',
                            raw_text, re.IGNORECASE)
    if label_match:
        return label_match.group(1).upper()
    # Fallback: find the bare token anywhere in the text.
    token_match = re.search(r"\b(EASY|NON-NESTED|NESTED)\b",
                            raw_text, re.IGNORECASE)
    return token_match.group(1).upper() if token_match else "EASY"


def _extract_sub_questions(raw_text: str) -> list[str]:
    """Pull decomposition sub-questions out of a stage-2 response.

    The classification demos for NESTED queries teach the model to emit

        sub_questions:
          Q1: "..."
          Q2: "..."

    We capture every `Q<digit>:` line. If none are found we return [] so
    stage 3 falls back to its class-specific non-decomposed template.
    """
    matches = re.findall(r'Q\d+\s*:\s*"([^"]+)"', raw_text)
    if matches:
        return matches
    # Permissive fallback: unquoted form "Q1: what is ..." up to newline.
    matches = re.findall(r"Q\d+\s*:\s*(.+)", raw_text)
    return [m.strip() for m in matches if m.strip()]


def _extract_tables_to_join(raw_text: str) -> list[str]:
    """Pull `tables_to_join: [...]` from a stage-2 response. Empty on miss."""
    match = re.search(r"tables_to_join\s*:\s*\[([^\]]*)\]", raw_text, re.IGNORECASE)
    if not match:
        return []
    inside = match.group(1)
    # Accept both "['relations', 'events']" and quoted bareword forms.
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", inside)
    return toks


# ── Hallucination checker ─────────────────────────────────────────────────────

def check_join_hallucination(sql: str, allowed_joins: set) -> bool:
    found: list[str] = []
    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(sql, read="duckdb")
        for eq in parsed.find_all(exp.EQ):
            left, right = eq.left, eq.right
            if isinstance(left, exp.Column) and left.name.lower() == "relation_type":
                if isinstance(right, exp.Literal) and right.is_string:
                    found.append(str(right.this))
            if isinstance(right, exp.Column) and right.name.lower() == "relation_type":
                if isinstance(left, exp.Literal) and left.is_string:
                    found.append(str(left.this))
        for in_expr in parsed.find_all(exp.In):
            target = in_expr.this
            if isinstance(target, exp.Column) and target.name.lower() == "relation_type":
                for item in in_expr.expressions:
                    if isinstance(item, exp.Literal) and item.is_string:
                        found.append(str(item.this))
    except Exception:
        pass

    if not found:
        found.extend(
            re.findall(r"relation_type\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
        )
        for inside in re.findall(r"relation_type\s+IN\s*\(([^)]*)\)", sql, re.IGNORECASE):
            found.extend(re.findall(r"['\"]([^'\"]+)['\"]", inside))
    return any(rt not in allowed_joins for rt in found)


# ── Schema text builder ───────────────────────────────────────────────────────

def build_schema_text(catalog: dict, whitelist: list) -> str:
    lines = ["DATABASE SCHEMA (DuckDB SQL):", ""]
    for tbl in catalog["tables"]:
        lines.append(f"Table: {tbl['table_name']}  ({tbl['row_count']:,} rows)")
        for col in tbl["columns"]:
            null_note = f" [nullable]" if col["nullable"] else ""
            lines.append(f"  {col['name']}  {col['dtype']}{null_note}")
        if tbl["temporal_fields"]:
            lines.append(f"  [temporal: {', '.join(tbl['temporal_fields'])}]")
        lines.append("")
    lines.append("EVENT TYPES (events.event_type):")
    for et in catalog["event_types"]:
        lines.append(f"  '{et}'")
    lines.append("")
    lines.append("OBJECT TYPES (events.object_type / objects.object_type):")
    for ot in catalog["object_types"]:
        lines.append(f"  '{ot}'")
    lines.append("")
    lines.append("ALLOWED JOINS (relations.relation_type — never use other values):")
    for w in whitelist:
        lines.append(
            f"  '{w['relation_type']}'  "
            f"{w['from_object_type']} -> {w['to_object_type']}"
        )
    lines += ["", "DuckDB notes:",
              "  year(timestamp), month(timestamp) are valid functions.",
              "  date_diff('day', ts1, ts2) computes day difference.",
              "  Do NOT use INSERT, UPDATE, DELETE, DROP, or CREATE."]
    return "\n".join(lines)


def build_fewshot_text(benchmark_rows: list) -> str:
    examples = [r for r in benchmark_rows if r["qid"] in FEW_SHOT_QIDS]
    lines = ["SOLVED EXAMPLES:", ""]
    for r in examples:
        lines += [f"Question: {r['nl_question']}", f"SQL: {r['gold_sql']}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# B1: Zero-shot  —  Rajkumar et al. 2022 "Create Table + Select 3" prompt
# ──────────────────────────────────────────────────────────────────────────────
#
# Paper: "Evaluating the Text-to-SQL Capabilities of Large Language Models"
#         (arXiv:2204.00498, Rajkumar, Li, Bahdanau 2022).
#
# The paper compares six zero-shot prompt styles (Question, API Docs, Select 1/3/5/10,
# Create Table, Create Table + Select X). Their best-performing setup is
# "Create Table + Select 3" — 67% execution accuracy on Spider dev with Codex
# (Table 2, §3). The recipe has three parts per table:
#
#   (1) A CREATE TABLE statement with column names, column types, and
#       primary-key / foreign-key declarations.
#   (2) A SQL comment showing the first 3 rows, written as if the model had
#       just executed `SELECT * FROM table LIMIT 3`.
#   (3) The test question written as a SQL comment, followed by the token
#       "SELECT" to prime completion — no instruction preamble.
#
# We reproduce that recipe here for the OCEL schema. The result is passed
# unchanged to the LLM (no "you are a SQL expert" preamble), matching the paper.

# Pandas / OCEL dtype → DuckDB SQL type. The catalog currently uses the short
# Python labels "str", "datetime64[us]", "float64", "int64" / "int32", "bool".
# Rajkumar's Appendix C Figure 10 uses "int", "text" (SQLite style); we use
# DuckDB's equivalents. The paper states the choice of types is just a hint
# for the LLM so the minor dialect swap is harmless.
_DTYPE_TO_SQL: dict[str, str] = {
    "str":              "VARCHAR",
    "datetime64[us]":   "TIMESTAMP",
    "datetime64[ns]":   "TIMESTAMP",
    "float64":          "DOUBLE",
    "float32":          "REAL",
    "int64":            "BIGINT",
    "int32":            "INTEGER",
    "bool":             "BOOLEAN",
}


def _dtype_to_sql(dtype: str) -> str:
    """Map a pandas-style dtype string to a DuckDB SQL type name.

    Falls back to VARCHAR for anything unknown — safer than failing, since the
    LLM only uses the type as a hint, not for validation.
    """
    return _DTYPE_TO_SQL.get(dtype, "VARCHAR")


# OCEL-specific primary-key declarations. Derived from the benchmark's schema
# profile: event_id is globally unique (see configs/schema_catalog.json),
# object_id is unique within objects, and relations has a composite key.
# Hard-coded because the catalog does not encode PK metadata and these values
# are stable across OCEL rebuilds.
_PRIMARY_KEYS: dict[str, list[str]] = {
    "events":    ["event_id"],
    "objects":   ["object_id"],
    "relations": ["from_object_id", "to_object_id", "relation_type"],
}

# OCEL-specific foreign-key declarations. Derived from the OCEL 2.0 linking
# semantics: every event's object_id links to objects.object_id, and both
# ends of a relation likewise point at objects.object_id.
_FOREIGN_KEYS: dict[str, list[tuple[str, str, str]]] = {
    # table: [(local_col, target_table, target_col), ...]
    "events":    [("object_id", "objects", "object_id")],
    "relations": [
        ("from_object_id", "objects", "object_id"),
        ("to_object_id",   "objects", "object_id"),
    ],
}


def _build_create_table_block(table_info: dict) -> str:
    """Emit a DuckDB `CREATE TABLE` statement for one table in the catalog,
    in the exact format Rajkumar Figure 10 shows: column declarations with
    types, `primary key (...)` as the last constraint, and any
    `foreign key (col) references tbl(target_col)` clauses inline.

    Not-null info comes from the catalog's `nullable` flag. PK/FK metadata
    lives in the `_PRIMARY_KEYS` and `_FOREIGN_KEYS` module-level dicts
    because the catalog JSON doesn't encode them.
    """
    name = table_info["table_name"]
    inner_lines: list[str] = []
    for col in table_info["columns"]:
        sql_type  = _dtype_to_sql(col["dtype"])
        null_note = "" if col["nullable"] else " NOT NULL"
        inner_lines.append(f"        {col['name']} {sql_type}{null_note}")

    pk_cols = _PRIMARY_KEYS.get(name)
    if pk_cols:
        inner_lines.append(f"        primary key ({', '.join(pk_cols)})")

    for local_col, target_tbl, target_col in _FOREIGN_KEYS.get(name, []):
        inner_lines.append(
            f"        foreign key ({local_col}) references {target_tbl}({target_col})"
        )

    return f"CREATE TABLE {name}(\n" + ",\n".join(inner_lines) + "\n)"


def _format_sample_rows_block(table_name: str, rows: list[dict]) -> str:
    """Format 3 sample rows as a SQL block comment, Rajkumar Figure 10 style.

    The exact template (from the paper's figure):

        /*
        3 example rows:
        SELECT * FROM <table> LIMIT 3;
        col1   col2   col3
        v1     v2     v3
        ...
        */

    Columns are left-aligned with two spaces between them, matching the
    visual style in the paper. Empty input → empty string.
    """
    if not rows:
        return ""
    cols = list(rows[0].keys())
    # Column widths = max(len(col_name), max(len(str(val)) for val in col)).
    widths = []
    for c in cols:
        max_val_len = max((len(str(r[c])) if r[c] is not None else 4) for r in rows)
        widths.append(max(len(c), max_val_len))

    def fmt_row(values: list[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    header = fmt_row(cols)
    body_lines = [fmt_row([str(r[c]) if r[c] is not None else "NULL" for c in cols])
                  for r in rows]
    return (
        "/*\n"
        "3 example rows:\n"
        f"SELECT * FROM {table_name} LIMIT 3;\n"
        + header + "\n"
        + "\n".join(body_lines) + "\n"
        "*/"
    )


def build_rajkumar_schema_prompt(
    catalog: dict,
    whitelist: list,  # kept in signature for API stability; not used directly
    sample_rows: dict[str, list[dict]] | None = None,
) -> str:
    """Build the full 'Create Table + Select 3' schema prefix.

    Layout matches Rajkumar 2022 Figure 10 exactly:

        CREATE TABLE <t1>(... primary key (...), foreign key (...) references ...(...))
        /*
        3 example rows:
        SELECT * FROM <t1> LIMIT 3;
        ...
        */

        CREATE TABLE <t2>(...)
        /*
        3 example rows:
        ...
        */

    Note that the sample-row comment sits *immediately* after its CREATE TABLE
    (no blank line between) while two tables are separated by a single blank
    line. When `sample_rows` is missing the block degrades to Rajkumar's
    "Create Table" variant (no Select 3).

    The `whitelist` parameter is accepted for backward compatibility only;
    Rajkumar's prompt embeds FK declarations inside each CREATE TABLE rather
    than in a separate comment block.
    """
    sample_rows = sample_rows or {}
    table_blocks: list[str] = []
    for tbl in catalog["tables"]:
        create_block = _build_create_table_block(tbl)
        rows_block   = _format_sample_rows_block(
            tbl["table_name"], sample_rows.get(tbl["table_name"], []),
        )
        # Sample-row comment is glued directly under the CREATE TABLE (paper's
        # Figure 10: the `/*` line immediately follows the `)` of CREATE TABLE).
        if rows_block:
            table_blocks.append(f"{create_block}\n{rows_block}")
        else:
            table_blocks.append(create_block)
    # Tables separated by a single blank line.
    return "\n\n".join(table_blocks)


def prompt_b1_rajkumar(question: str, schema_prefix: str) -> str:
    """Build the final B1 prompt, matching Rajkumar 2022 Figure 10.

    The paper's tail is exactly:

        -- Using valid SQLite, answer the following questions for the tables provided above.

        -- What is Kyle's id?
        SELECT

    We substitute "DuckDB" for "SQLite" since the OCEL benchmark executes
    on DuckDB. We also add a single-line instruction that tells chat-tuned
    LLMs (DeepSeek-chat, GPT-4-Turbo, Claude) to emit pure SQL rather than
    natural-language commentary. The original paper used completion-style
    Codex/GPT-3, which obeys the bare "SELECT" priming; chat models need
    the explicit instruction or they explain what they would write.
    """
    return (
        f"{schema_prefix}\n\n"
        f"-- Using valid DuckDB, answer the following question for the tables provided above.\n"
        f"-- Output only a single SQL query starting with SELECT — no markdown, no explanation.\n"
        f"\n"
        f"-- {question}\n"
        f"SELECT"
    )


def fetch_sample_rows_duckdb(
    ocel_dir,
    n: int = 3,
) -> dict[str, list[dict]]:
    """Fetch `n` sample rows per OCEL table from the parquet files.

    This is a standalone helper so callers (`run_full_eval.py`, notebooks)
    can compute sample rows once and reuse the dict across translator
    instances. Uses its own DuckDB connection and closes it after the reads.

    Parameters
    ----------
    ocel_dir : str or Path
        Directory containing `events.parquet`, `objects.parquet`,
        `relations.parquet`.
    n : int, default 3
        Number of sample rows per table (Rajkumar's "Select 3").

    Returns
    -------
    dict[str, list[dict]]
        Keys: "events", "objects", "relations". Values: list of row-dicts.
    """
    import duckdb
    from pathlib import Path as _Path
    ocel_path_str = str(_Path(ocel_dir)).replace("\\", "/").replace("'", "''")

    conn = duckdb.connect()
    try:
        out: dict[str, list[dict]] = {}
        for table_name in ("events", "objects", "relations"):
            sql = (
                f"SELECT * FROM read_parquet('{ocel_path_str}/{table_name}.parquet') "
                f"LIMIT {int(n)}"
            )
            df = conn.execute(sql).df()
            # Convert to list-of-dicts with plain Python scalars so downstream
            # callers (prompt builders) don't have to handle numpy/pandas types.
            out[table_name] = df.astype(object).where(df.notna(), None).to_dict("records")
        return out
    finally:
        conn.close()


# Keep the old name importable as an alias, so any external notebook code
# that did `from nl2ocel.baseline_translator import prompt_b1` still works.
def prompt_b1(question: str, schema_prefix: str = "") -> str:
    """Backward-compatible B1 entry-point. Prefer `prompt_b1_rajkumar`.

    If `schema_prefix` is empty we fall back to the bare "Question-only"
    variant from Rajkumar §3 (their weakest zero-shot baseline, 8.3% EX on
    Spider) — only useful for smoke tests.
    """
    if schema_prefix:
        return prompt_b1_rajkumar(question, schema_prefix)
    return (
        f"-- Using valid DuckDB SQL, answer the following question.\n"
        f"-- The database contains tables: events, objects, relations.\n"
        f"-- Question: {question}\n"
        f"SELECT"
    )


# ──────────────────────────────────────────────────────────────────────────────
# B3: DIN-SQL 4-stage prompts  —  Pourreza & Rafiei, NeurIPS 2023
# ──────────────────────────────────────────────────────────────────────────────
#
# Paper: "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with
#         Self-Correction" (NeurIPS 2023).
#
# Implementation follows §4.1-4.4. Every stage is few-shot: 10 demos for
# schema linking and classification, 5 demos per class for SQL generation.
# Self-correction uses the "gentle" or "generic" prompt depending on the
# LLM family (§4.4: gentle for GPT-4 family, generic for CodeX family).
#
# Demo bank lives in configs/din_sql_demos/demos.json; renderers in
# src/nl2ocel/din_sql_demos.py.
#
# DIN-SQL §5.2 hyperparameters:
#   - Greedy decoding, temperature = 0                (handled in llm_client)
#   - max_tokens = 600 for stages 1-3, 350 for stage 4 (passed per-call)
#   - stop = ["Q:"] for stages 1-3, ["#;\n\n"] for stage 4

# ---- Stage 1: Schema Linking (DIN-SQL §4.1) --------------------------------

def prompt_din_sql_stage1_schema_linking(question: str) -> str:
    """Build the DIN-SQL stage-1 schema-linking prompt.

    The full prompt is: 10 demonstrations (each showing schema + question +
    CoT reasoning + final `Schema_links:` line), then the test schema summary
    and the test question primed with "Let's think step by step." — matching
    Figure 3(a) and Appendix A.3 of the paper.
    """
    from .din_sql_demos import render_schema_linking_demos, din_sql_schema_summary
    demos = render_schema_linking_demos(k=10)
    return (
        f"{demos}\n\n####\n\n"
        f"{din_sql_schema_summary()}\n"
        f"Q: \"{question}\"\n"
        f"A: Let's think step by step."
    )


# ---- Stage 2: Classification & Decomposition (DIN-SQL §4.2) -----------------

def prompt_din_sql_stage2_classify(question: str, schema_links: str) -> str:
    """Build the DIN-SQL stage-2 classification prompt.

    Format (Figure 3(b), Appendix A.4): 10 demos each showing question +
    schema_links + CoT + final `Label: "EASY"|"NON-NESTED"|"NESTED"`, then
    the test question with its schema_links (from stage 1) and a bare
    "A: Let's think step by step." to prime the CoT.
    """
    from .din_sql_demos import render_classification_demos
    demos = render_classification_demos(k=10)
    return (
        f"{demos}\n\n####\n\n"
        f"Q: \"{question}\"\n"
        f"schema_links: {schema_links}\n"
        f"A: Let's think step by step."
    )


# ---- Stage 3: SQL Generation (DIN-SQL §4.3) --------------------------------

def prompt_din_sql_stage3_generate(
    question: str,
    schema_links: str,
    classification: str,
    sub_questions: list[str] | None = None,
) -> str:
    """Build the DIN-SQL stage-3 SQL generation prompt (class-specific).

    DIN-SQL uses a different prompt template per class (§4.3):
      - EASY        : `<Q, schema_links, SQL>`
      - NON-NESTED  : `<Q, schema_links, NatSQL, SQL>`
      - NESTED      : `<Q, schema_links, Q1, A1, ..., Qk, Ak, NatSQL, SQL>`

    The test-question tail uses the same template as the class-matched demos
    so the LLM's continuation extends the `SQL:` line.

    Parameters
    ----------
    question : str
        Natural-language test question.
    schema_links : str
        Stage-1 output — the relevant subset of the schema, formatted as the
        `Schema_links: [...]` line.
    classification : str
        Stage-2 class label (EASY / NON-NESTED / NESTED).
    sub_questions : list[str], optional
        Stage-2 decomposition sub-questions (NESTED only). When supplied,
        each appears as a `Q_i:` line in the test-question tail so the LLM
        is prompted to solve them in order before producing the final SQL.
        Ignored for EASY / NON-NESTED classes.
    """
    from .din_sql_demos import render_sql_gen_demos

    cls = classification.strip().upper()
    if "NESTED" in cls and "NON" not in cls:
        class_label = "NESTED"
        tail_lines = [
            f"Q: \"{question}\"",
            f"schema_links: {schema_links}",
        ]
        if sub_questions:
            for i, sq in enumerate(sub_questions, start=1):
                tail_lines.append(f"Q{i}: \"{sq}\"")
                tail_lines.append(f"A{i}:")
        tail_lines.append("NatSQL:")
        tail_lines.append("SQL:")
        tail = "\n".join(tail_lines)
    elif "NON-NESTED" in cls:
        class_label = "NON-NESTED"
        tail = (
            f"Q: \"{question}\"\n"
            f"schema_links: {schema_links}\n"
            f"NatSQL:\n"
            f"SQL:"
        )
    else:  # EASY (default fall-through)
        class_label = "EASY"
        tail = (
            f"Q: \"{question}\"\n"
            f"schema_links: {schema_links}\n"
            f"SQL:"
        )

    demos = render_sql_gen_demos(class_label, k=5)
    return f"{demos}\n\n####\n\n{tail}"


# ---- Stage 4: Self-Correction (DIN-SQL §4.4) --------------------------------
#
# The paper defines TWO correction prompts:
#   - generic  : "There is a bug — fix it."             (default for CodeX)
#   - gentle   : "Check for these specific issues..."   (default for GPT-4)
# We expose both and let the caller pick via the model name.

def prompt_din_sql_stage4_generic(generated_sql: str) -> str:
    """DIN-SQL §4.4 generic self-correction prompt.

    Asserts the SQL is buggy and asks for the fix. This was empirically the
    best correction strategy for CodeX in the paper (Table 5). We also use
    it for DeepSeek since DeepSeek-chat is closer to CodeX than GPT-4 in
    both training regime and observed behavior.
    """
    return (
        f"### Buggy SQL:\n{generated_sql}\n\n"
        f"### Fixed SQL:\n"
    )


def prompt_din_sql_stage4_gentle(generated_sql: str, schema_summary: str) -> str:
    """DIN-SQL §4.4 gentle self-correction prompt.

    Does NOT assume the SQL is buggy; asks the model to check for a list of
    common issues and return the query unchanged if no issue is found. This
    is the DIN-SQL default for GPT-4 because GPT-4 tends to over-correct
    when told the SQL is buggy.
    """
    return (
        f"{schema_summary}\n\n"
        f"### SQL query to review:\n{generated_sql}\n\n"
        f"Check the SQL above for these potential issues:\n"
        f"  1. Wrong or hallucinated table/column names (must match schema).\n"
        f"  2. Invalid relation_type values (must be from the allowed list).\n"
        f"  3. Incorrect JOIN conditions.\n"
        f"  4. Wrong event_type or object_type string literals.\n"
        f"  5. Incorrect DuckDB function syntax.\n\n"
        f"If the SQL is already correct, return it unchanged.\n"
        f"Otherwise return only the corrected SQL, no explanation.\n\n"
        f"### Final SQL:\n"
    )


def pick_self_correction_prompt(model: str) -> str:
    """Return 'generic' or 'gentle' depending on the model family.

    DIN-SQL §4.4 recipe:
      - GPT-4 family  -> gentle
      - CodeX family  -> generic
      - Everything else: we use `generic` by default, matching CodeX since
        open-weight / DeepSeek models behave closer to CodeX than GPT-4.
    """
    name = (model or "").lower()
    if "gpt-4" in name or "gpt4" in name:
        return "gentle"
    return "generic"


# ── B2: Few-shot (Nan 2023) ──────────────────────────────────────────────────
# Legacy static prompt helper retained for API compatibility. The reported B2
# path delegates to NanFewShotPipeline for similarity-diversity demonstration
# selection, schema augmentation, and shot-count voting.

def prompt_b2_fewshot(question: str, schema_text: str, fewshot_text: str) -> str:
    """
    Legacy prompt formatter used by older callers.

    Production few-shot evaluation is handled by NanFewShotPipeline so the
    benchmarked B2 baseline remains faithful to Nan 2023.
    """
    return (
        "You are a SQL expert. Write a single DuckDB SQL query that answers the question.\n"
        "Return ONLY the SQL query. No explanation.\n"
        "Do NOT use INSERT, UPDATE, DELETE, DROP, or CREATE.\n\n"
        f"{schema_text}\n\n"
        f"{fewshot_text}\n"
        f"Question: {question}\n\nSQL:"
    )


# ── Main translator class ─────────────────────────────────────────────────────

class BaselineTranslator:
    # Canonical mode names match manuscript labels B1/B2/B3.
    MODES = ("zero_shot", "few_shot", "din_sql")

    def __init__(self, mode: str, catalog: dict, whitelist: list,
                 benchmark_rows: Optional[list] = None,
                 sample_rows: Optional[dict] = None,
                 sql_executor=None,
                 nan_shot_range=range(4, 11),
                 model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL,
                 backend: str = "ollama", api_key: str | None = None):
        """Construct a baseline translator.

        Parameters
        ----------
        mode : str
            One of `zero_shot` (B1, Rajkumar 2022), `few_shot` (B2, Nan 2023),
            `din_sql` (B3, Pourreza 2023).
        catalog : dict
            Parsed `configs/schema_catalog.json`.
        whitelist : list[dict]
            Parsed `configs/relation_whitelist.json["whitelist"]`.
        benchmark_rows : list, optional
            Annotated pool for Nan 2023's Similarity-Diversity sampler. Each
            row must have keys `qid`, `nl_question`, `gold_sql`. For B2 the
            three example QIDs (Q001/Q019/Q025) used at demo-design time in
            the legacy implementation are still excluded at eval time to
            avoid data leakage.
        sample_rows : dict, optional
            Map from table name to list of sample-row dicts. Used by the
            Rajkumar 'Create Table + Select 3' prompt for B1. Callers should
            populate this with `fetch_sample_rows_duckdb(ocel_dir)` before
            instantiation; if omitted, B1 falls back to 'Create Table' only.
        sql_executor : callable (str -> tuple[bool, str]), optional
            Runs a candidate SQL and returns (exec_ok, result_hash). Required
            only for mode="few_shot" because Nan 2023's voting loop (Algorithm 2)
            removes exec-error shots before majority-voting on result hashes.
            `run_full_eval.py` provides an executor backed by the same DuckDB
            connection it uses for denotation-accuracy scoring.
        nan_shot_range : iterable of int, optional
            Shot counts to sweep in the Nan voting loop. Paper default is
            4..10 inclusive. Can be narrowed to reduce LLM cost.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode          = mode
        self.model         = model
        self.ollama_url    = ollama_url
        self.backend       = backend   # "ollama" | "deepseek"
        self.api_key       = api_key
        self.allowed_joins = {w["relation_type"] for w in whitelist}

        # Legacy flat schema text — kept for any downstream callers that still
        # import `build_schema_text` (no longer used by B1/B2/B3).
        self._schema_text  = build_schema_text(catalog, whitelist)
        self._fewshot_text = build_fewshot_text(benchmark_rows or [])

        # Rajkumar 'Create Table + Select 3' prefix for B1. Built once so
        # every question reuses the same schema block (no LLM-side caching
        # needed; the prompt text itself is identical across questions).
        self._rajkumar_schema = build_rajkumar_schema_prompt(
            catalog, whitelist, sample_rows=sample_rows
        )

        # Nan 2023 integrated pipeline (only needed for mode="few_shot").
        # Built lazily because constructing it runs k-Means on the pool.
        self._nan_pipeline = None
        self._sql_executor = sql_executor
        self._nan_shot_range = list(nan_shot_range)
        if mode == "few_shot":
            from .nan_sampler import NanFewShotPipeline
            pool = [
                {
                    "qid":          r["qid"],
                    "nl_question":  r["nl_question"],
                    "gold_sql":     r["gold_sql"],
                    "difficulty":   r.get("difficulty", ""),
                }
                for r in (benchmark_rows or [])
                if r.get("qid") not in FEW_SHOT_QIDS  # historical exclusion
            ]
            self._nan_pipeline = NanFewShotPipeline(
                pool_rows=pool, catalog=catalog, whitelist=whitelist, k=10,
            )

    def _llm(
        self,
        prompt: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        """Single LLM call with optional per-call hyperparameters.

        `max_tokens` and `stop` are forwarded to the backend — used by the
        DIN-SQL stages to match §5.2 (stages 1-3: max_tokens=600, stop=['Q:'];
        stage 4: max_tokens=350, stop=['#;\\n\\n']).
        """
        if self.backend == "deepseek":
            return call_deepseek(
                prompt, model=self.model, api_key=self.api_key,
                max_tokens=max_tokens, stop=stop,
            )
        return call_ollama(
            prompt, model=self.model, ollama_url=self.ollama_url,
            max_tokens=max_tokens, stop=stop,
        )

    def translate(self, question: str, qid: str | None = None) -> dict:
        """Translate one question to predicted SQL.

        Parameters
        ----------
        question : str
            Natural-language question.
        qid : str, optional
            Question id from the benchmark. When provided, the few-shot (B2)
            mode passes it as `exclude_qid` so the Nan sampler skips this
            row in its own demonstration pool — the leave-one-out guard for
            dev-set evaluation. B1 and B3 ignore the parameter.
        """
        t_total = time.time()
        stages  = {}

        if self.mode == "zero_shot":
            # B1 uses Rajkumar 2022 "Create Table + Select 3" (Figure 10).
            # §A.1 hyperparameters: max_tokens=200 and stop tokens that end
            # a SQL statement when the LLM starts writing a new one.
            RAJKUMAR_MAX_TOKENS = 200
            RAJKUMAR_STOP       = ["--", "\n\n", ";", "#"]

            out = self._llm(
                prompt_b1_rajkumar(question, self._rajkumar_schema),
                max_tokens=RAJKUMAR_MAX_TOKENS, stop=RAJKUMAR_STOP,
            )
            pred_sql = extract_sql(out["text"]) if not out["error"] else ""
            # The prompt primes with "SELECT" — LLM may return either a full
            # query starting with SELECT or just the continuation after it.
            # Repair by prepending "SELECT " when needed.
            if pred_sql and not re.match(r"^\s*(select|with)\b", pred_sql, re.IGNORECASE):
                pred_sql = "SELECT " + pred_sql
            stages = {"stage1": out}

        elif self.mode == "din_sql":
            # DIN-SQL §5.2 hyperparameters (Pourreza & Rafiei 2023).
            # Stages 1-3 use max_tokens=600 and stop=["Q:"] because every demo
            # example in the prompt begins the next block with "Q:"; hitting it
            # means the model is trying to hallucinate a new demo. Stage 4 uses
            # max_tokens=350 and stop=["#;\n\n"] because corrected SQL is short.
            MAX_TOKENS_STAGES_123 = 600
            MAX_TOKENS_STAGE_4    = 350
            STOP_STAGES_123       = ["Q:"]
            STOP_STAGE_4          = ["#;\n\n"]

            from .din_sql_demos import din_sql_schema_summary
            schema_summary = din_sql_schema_summary()

            # ── Stage 1: Schema linking  (§4.1, demo bank k=10) ──────────────
            s1 = self._llm(
                prompt_din_sql_stage1_schema_linking(question),
                max_tokens=MAX_TOKENS_STAGES_123, stop=STOP_STAGES_123,
            )
            schema_links = _extract_schema_links(s1["text"]) if not s1["error"] else "[]"

            # ── Stage 2: Classification + Decomposition  (§4.2, k=10) ────────
            # Per paper §4.2, stage 2 also detects (a) tables to be joined for
            # non-nested + nested queries and (b) sub-questions for nested
            # queries. We parse those out of the LLM response and thread them
            # into stage 3.
            s2 = self._llm(
                prompt_din_sql_stage2_classify(question, schema_links),
                max_tokens=MAX_TOKENS_STAGES_123, stop=STOP_STAGES_123,
            )
            classification_raw = s2["text"].strip() if not s2["error"] else "EASY"
            classification     = _extract_class_label(classification_raw)
            stage2_sub_questions = _extract_sub_questions(classification_raw)
            stage2_tables_to_join = _extract_tables_to_join(classification_raw)

            # ── Stage 3: SQL generation  (§4.3, class-specific demos k=5) ────
            # For NESTED we pass the sub-questions extracted above so the
            # prompt mirrors the demo format <Q, S, Q_1, A_1, ..., Q_k, A_k, I, A>.
            s3 = self._llm(
                prompt_din_sql_stage3_generate(
                    question, schema_links, classification,
                    sub_questions=stage2_sub_questions if classification == "NESTED" else None,
                ),
                max_tokens=MAX_TOKENS_STAGES_123, stop=STOP_STAGES_123,
            )
            raw_sql = extract_sql(s3["text"]) if not s3["error"] else ""

            # ── Stage 4: Self-correction  (§4.4, generic vs gentle per model) ─
            correction_mode = pick_self_correction_prompt(self.model)
            if raw_sql:
                if correction_mode == "generic":
                    s4_prompt = prompt_din_sql_stage4_generic(raw_sql)
                else:
                    s4_prompt = prompt_din_sql_stage4_gentle(raw_sql, schema_summary)
                s4 = self._llm(
                    s4_prompt, max_tokens=MAX_TOKENS_STAGE_4, stop=STOP_STAGE_4,
                )
                pred_sql = extract_sql(s4["text"]) if not s4["error"] else raw_sql
            else:
                s4 = {"text": "", "latency_s": 0.0, "error": "no SQL from stage 3"}
                pred_sql = raw_sql

            stages = {
                "stage1_schema_linking": s1,
                "stage2_classification": {"text": classification, "raw": classification_raw,
                                          "latency_s": s2["latency_s"], "error": s2["error"]},
                "stage3_sql_gen":        s3,
                "stage4_self_correct":   {**s4, "mode": correction_mode},
            }

        else:  # few_shot (Nan 2023 — full Integrated Strategy, Algorithm 2)
            # The Nan pipeline requires:
            #   (1) an initial predictor for the draft SQL — we reuse B1's
            #       Rajkumar 'Create Table + Select 3' prompt so the draft
            #       already contains schema hints.
            #   (2) a final predictor — the LLM with the few-shot prompt.
            #   (3) an executor — injected at construction time so voting
            #       can discard exec-error candidates.
            if self._nan_pipeline is None:
                raise RuntimeError("few_shot mode requires benchmark_rows to build the Nan pool.")
            if self._sql_executor is None:
                raise RuntimeError("few_shot mode requires sql_executor (see run_full_eval.py).")

            def _initial(q: str) -> str:
                """Zero-shot draft used only for difficulty routing (cheap)."""
                out = self._llm(prompt_b1_rajkumar(q, self._rajkumar_schema))
                draft = extract_sql(out["text"]) if not out["error"] else ""
                if draft and not re.match(r"^\s*(select|with)\b", draft, re.IGNORECASE):
                    draft = "SELECT " + draft
                return draft

            def _final(prompt_text: str) -> dict:
                """Full few-shot prediction; return dict for _coerce_sql."""
                return self._llm(prompt_text)

            vote_out = self._nan_pipeline.predict_with_voting(
                question=question,
                initial_predictor=_initial,
                final_predictor=_final,
                executor=self._sql_executor,
                shot_range=self._nan_shot_range,
                exclude_qid=qid,   # leave-one-out guard
            )
            pred_sql = vote_out["pred_sql"]
            stages = {
                "nan_pipeline": {
                    "text":      pred_sql,
                    "error":     None if pred_sql else "no SQL after voting",
                    "latency_s": sum(s["latency_s"] for s in vote_out["per_shot"]),
                    "per_shot":  vote_out["per_shot"],
                    "draft_sql": vote_out["draft_sql"],
                    "difficulty":vote_out["difficulty"],
                    "vote_hash": vote_out["vote_hash"],
                },
            }

        total_latency = round(time.time() - t_total, 3)
        llm_error     = next((v["error"] for v in stages.values() if v.get("error")), None)
        join_hall     = check_join_hallucination(pred_sql, self.allowed_joins) if pred_sql else False

        return {
            "mode":          self.mode,
            "pred_sql":      pred_sql,
            "latency_s":     total_latency,
            "join_hall":     join_hall,
            "llm_error":     llm_error or "",
            "stages":        stages,
        }


# ── CLI eval runner ───────────────────────────────────────────────────────────

def run_eval(mode: str, benchmark_path: Path, catalog_path: Path,
             whitelist_path: Path, ocel_dir: Path, out_path: Path,
             model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL,
             subset: Optional[list] = None) -> list:
    import duckdb

    catalog   = json.loads(catalog_path.read_text(encoding="utf-8"))
    whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))["whitelist"]

    with open(benchmark_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if subset:
        rows = [r for r in rows if r["qid"] in subset]

    # Exclude the 3 questions used as fixed few-shot demonstrations from
    # evaluation when running the few-shot baseline (avoids data leakage).
    exclude    = FEW_SHOT_QIDS if mode == "few_shot" else []
    eval_rows  = [r for r in rows if r["qid"] not in exclude]

    conn = duckdb.connect()
    _ocel = str(ocel_dir).replace("\\", "/").replace("'", "''")
    conn.execute(f"CREATE VIEW events    AS SELECT * FROM read_parquet('{_ocel}/events.parquet')")
    conn.execute(f"CREATE VIEW objects   AS SELECT * FROM read_parquet('{_ocel}/objects.parquet')")
    conn.execute(f"CREATE VIEW relations AS SELECT * FROM read_parquet('{_ocel}/relations.parquet')")

    def _hash(sql):
        try:
            df = conn.execute(sql).df()
            return hash_dataframe(df), None
        except Exception as e:
            return None, str(e)

    def _executor(sql):
        result_hash, err = _hash(sql)
        return err is None, result_hash or ""

    translator = BaselineTranslator(
        mode=mode,
        catalog=catalog,
        whitelist=whitelist,
        benchmark_rows=rows,
        sql_executor=_executor if mode == "few_shot" else None,
        model=model,
        ollama_url=ollama_url,
    )

    results = []
    try:
        for i, row in enumerate(eval_rows, 1):
            out       = translator.translate(row["nl_question"], qid=row["qid"])
            pred_sql  = out["pred_sql"]
            pred_hash, exec_err = _hash(pred_sql) if pred_sql else (None, "no SQL")
            exec_ok   = exec_err is None
            den_acc   = (pred_hash == row["gold_result_hash"]) if exec_ok else False

            results.append({
                "method":      mode,
                "qid":         row["qid"],
                "query_class": row["query_class"],
                "difficulty":  row["difficulty"],
                "nl_question": row["nl_question"],
                "pred_sql":    pred_sql,
                "exec_ok":     exec_ok,
                "den_acc":     den_acc,
                "join_hall":   out["join_hall"],
                "latency_s":   out["latency_s"],
                "exec_error":  exec_err or "",
                "llm_error":   out["llm_error"],
            })

            status = "OK" if den_acc else ("EXEC_FAIL" if not exec_ok else "WRONG")
            print(f"[{i:02d}/{len(eval_rows)}] {row['qid']} ({row['query_class']}/{row['difficulty']}) "
                  f"-> {status}  ({out['latency_s']:.1f}s)")
    finally:
        conn.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    n = len(results)
    print(f"\nMode={mode}  N={n}  "
          f"ExecRate={sum(r['exec_ok'] for r in results)/n:.3f}  "
          f"DenAcc={sum(r['den_acc'] for r in results)/n:.3f}  "
          f"JoinHallRate={sum(r['join_hall'] for r in results)/n:.3f}")
    print(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run one NL-to-SQL baseline on OCEL benchmark")
    p.add_argument("--mode",       required=True, choices=BaselineTranslator.MODES)
    p.add_argument("--benchmark",  required=True)
    p.add_argument("--catalog",    required=True)
    p.add_argument("--whitelist",  required=True)
    p.add_argument("--ocel_dir",   required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--ollama_url", default=OLLAMA_URL)
    args = p.parse_args()

    run_eval(mode=args.mode, benchmark_path=Path(args.benchmark),
             catalog_path=Path(args.catalog), whitelist_path=Path(args.whitelist),
             ocel_dir=Path(args.ocel_dir), out_path=Path(args.out),
             model=args.model, ollama_url=args.ollama_url)
