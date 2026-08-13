"""
din_sql_demos.py — render DIN-SQL few-shot demonstrations from the master JSON.

Loads `configs/din_sql_demos/demos.json` once and exposes one helper per
DIN-SQL stage:

  - render_schema_linking_demos()   -> str
  - render_classification_demos()   -> str
  - render_sql_gen_demos(cls)       -> str   for cls in EASY / NON-NESTED / NESTED
  - render_self_correction_demos()  -> str   (not required by the paper; empty)

Each helper returns a ready-to-inject text block in the format DIN-SQL
Figure 3 (Appendix A.3-A.6) shows — one demo per `####`-separated section.
`BaselineTranslator` prepends the demo block to every stage prompt, matching
Pourreza 2023 §4.1-4.4.

Demo pool assignment (from `demos.json`):

  Stage               | Demos used             | Count
  --------------------|------------------------|------
  Schema linking      | all 15 demos, class-mixed | 10 (paper spec)
  Classification      | all 15 demos, class-mixed | 10 (paper spec)
  SQL gen / easy      | 5 EASY demos              | 5
  SQL gen / non-nest. | 5 NON-NESTED demos        | 5
  SQL gen / nested    | 5 NESTED demos            | 5

The paper uses 10 demos for stages 1-2 and a class-specific subset for stage 3.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

# Path resolves to <repo>/configs/din_sql_demos/demos.json. Keep module-level
# so callers don't have to pass it each time.
_DEMOS_PATH = Path(__file__).resolve().parents[2] / "configs" / "din_sql_demos" / "demos.json"


def _load_demos() -> list[dict]:
    """Read the master demo JSON and return its demos list. Cached via module
    globals so repeated calls don't hit the disk.
    """
    global _CACHED_DEMOS
    try:
        return _CACHED_DEMOS  # type: ignore[name-defined]
    except NameError:
        pass
    data = json.loads(_DEMOS_PATH.read_text(encoding="utf-8"))
    _CACHED_DEMOS = data["demos"]
    return _CACHED_DEMOS


# ─────────────────────────────────────────────────────────────────────────────
# Schema used across all demo prompts — DIN-SQL uses a compact schema summary
# ("Table X, columns = [...]") rather than CREATE TABLE. We reproduce that
# format here; a single copy is used for every demo because every OCEL demo
# queries the same three tables.
# ─────────────────────────────────────────────────────────────────────────────

_DIN_SQL_SCHEMA_SUMMARY = """Table events, columns = [event_id, event_type, timestamp, object_id, object_type, source_table]
Table objects, columns = [object_id, object_type, FKDAT, NETWR, WAERK, ZTERM, KUNAG, BUKRS, AUGDT, WRBTR, WAERS, KUNNR, XBLNR, cleared]
Table relations, columns = [from_object_id, from_object_type, to_object_id, to_object_type, relation_type, source_table]
Foreign_keys = [events.object_id = objects.object_id, relations.from_object_id = objects.object_id, relations.to_object_id = objects.object_id]
Allowed relation_type values = ['order_to_customer', 'order_to_material', 'order_to_delivery', 'order_to_billing', 'delivery_to_billing', 'billing_to_ar', 'order_to_payer']"""


def din_sql_schema_summary() -> str:
    """Expose the compact DIN-SQL-style schema summary for callers that want
    it outside of demo rendering (e.g. the test-question prompt)."""
    return _DIN_SQL_SCHEMA_SUMMARY


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Schema Linking demos (paper spec: 10 demos, CoT reasoning)
# ─────────────────────────────────────────────────────────────────────────────

def render_schema_linking_demos(k: int = 10, seed: int = 42) -> str:
    """Build the schema-linking demo block.

    Each demo follows the DIN-SQL Figure 3(a) format:

        Table ... columns = ...
        Foreign_keys = ...
        Q: "<question>"
        A: Let's think step by step. <reasoning>
        So the Schema_links are:
        Schema_links: <links>

    Parameters
    ----------
    k : int
        Number of demos to render (paper uses 10).
    seed : int
        RNG seed so the demo order is deterministic across runs. DIN-SQL says
        examples are "randomly selected" — we fix the seed to keep evaluation
        reproducible.
    """
    demos = _load_demos()
    rng = random.Random(seed)
    selected = rng.sample(demos, k=min(k, len(demos)))

    blocks = []
    for demo in selected:
        sl = demo["schema_linking"]
        blocks.append(
            f"{_DIN_SQL_SCHEMA_SUMMARY}\n"
            f"Q: \"{demo['question']}\"\n"
            f"A: Let's think step by step. {sl['reasoning']}\n"
            f"So the Schema_links are:\n"
            f"Schema_links: {sl['schema_links']}"
        )
    return "\n\n####\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Classification & Decomposition demos (paper spec: 10 demos)
# ─────────────────────────────────────────────────────────────────────────────

def render_classification_demos(k: int = 10, seed: int = 42) -> str:
    """Build the classification demo block (DIN-SQL §4.2 / Figure 3b).

    Format:
      - EASY        : Q + schema_links + CoT + Label
      - NON-NESTED  : Q + schema_links + CoT + tables_to_join + Label
      - NESTED      : Q + schema_links + CoT + tables_to_join + decomposition
                      sub-questions + Label
    This matches the paper: "query classification and decomposition also
    detects the set of tables to be joined for both non-nested and nested
    complex queries as well as any sub-queries that may be detected for
    nested queries" (§4.2).
    """
    demos = _load_demos()
    rng = random.Random(seed + 1)  # offset so stage-2 shuffle differs from stage-1
    selected = rng.sample(demos, k=min(k, len(demos)))

    blocks = []
    for demo in selected:
        sl  = demo["schema_linking"]
        cls = demo["classification"]
        gen = demo.get("sql_generation", {}) or {}
        label = cls["label"]

        body_lines = [
            f"Q: \"{demo['question']}\"",
            f"schema_links: {sl['schema_links']}",
            f"A: Let's think step by step. {cls['reasoning']}",
        ]

        # Both NON-NESTED and NESTED demos show the tables the model should join.
        tables_to_join = cls.get("tables_to_join")
        if tables_to_join:
            body_lines.append(f"tables_to_join: {tables_to_join}")

        # NESTED demos also expose the sub-questions so the model learns to
        # emit them — stage 3 will consume them when generating the final SQL.
        if label == "NESTED":
            sub_queries = gen.get("sub_queries") or []
            if sub_queries:
                body_lines.append("sub_questions:")
                for i, sq in enumerate(sub_queries, 1):
                    body_lines.append(f"  Q{i}: \"{sq['question']}\"")

        body_lines.append(f"Label: \"{label}\"")
        blocks.append("\n".join(body_lines))
    return "\n\n####\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: SQL Generation demos — class-specific (DIN-SQL §4.3)
# ─────────────────────────────────────────────────────────────────────────────

def _demos_for_class(cls_label: str) -> list[dict]:
    """Return all demos whose complexity field matches `cls_label` (case-insensitive)."""
    cls_label = cls_label.upper().strip()
    return [d for d in _load_demos() if d["complexity"].upper() == cls_label]


def render_sql_gen_demos(cls_label: str, k: int = 5, seed: int = 42) -> str:
    """Render class-specific SQL generation demos for DIN-SQL stage 3.

    - EASY        -> format <Q, S, A> from DIN-SQL §4.3 first paragraph.
    - NON-NESTED  -> format <Q, S, I, A> (NatSQL intermediate between S and A).
    - NESTED      -> format <Q, S, Q_1, A_1, ..., Q_n, A_n, I, A>
                     with hand-authored sub-questions.

    Parameters
    ----------
    cls_label : str
        "EASY", "NON-NESTED", or "NESTED" (case-insensitive).
    k : int
        Max demos to render. Paper uses 5-ish per class (Appendix A.4 gives
        "several examples per class").
    seed : int
        RNG seed.
    """
    demos = _demos_for_class(cls_label)
    if not demos:
        return ""
    rng = random.Random(seed + hash(cls_label) % 100)
    selected = rng.sample(demos, k=min(k, len(demos)))

    blocks = []
    for demo in selected:
        sl  = demo["schema_linking"]
        gen = demo["sql_generation"]
        cls = cls_label.upper()

        if cls == "EASY":
            # <Q, S, A>
            blocks.append(
                f"Q: \"{demo['question']}\"\n"
                f"schema_links: {sl['schema_links']}\n"
                f"SQL: {gen['sql']}"
            )
        elif cls == "NON-NESTED":
            # <Q, S, I, A> — NatSQL IR sits between schema_links and the final SQL.
            natsql = gen.get("natsql", "")
            blocks.append(
                f"Q: \"{demo['question']}\"\n"
                f"schema_links: {sl['schema_links']}\n"
                f"NatSQL: {natsql}\n"
                f"SQL: {gen['sql']}"
            )
        else:  # NESTED
            # Paper format (§4.3 last paragraph):
            #   <Q_j, S_j, Q_{j_1}, A_{j_1}, ..., Q_{j_k}, A_{j_k}, I_j, A_j>
            # where each (Q_i, A_i) pair is a sub-question and its SQL, I_j
            # is a NatSQL intermediate representation that combines the
            # sub-results, and A_j is the final DuckDB SQL.
            sub_queries = gen.get("sub_queries", []) or []
            lines = [
                f"Q: \"{demo['question']}\"",
                f"schema_links: {sl['schema_links']}",
            ]
            for i, sq in enumerate(sub_queries, start=1):
                lines.append(f"Q{i}: \"{sq['question']}\"")
                lines.append(f"A{i}: {sq['sql']}")
            natsql = gen.get("natsql", "")
            if natsql:
                lines.append(f"NatSQL: {natsql}")
            lines.append(f"SQL: {gen['sql']}")
            blocks.append("\n".join(lines))
    return "\n\n####\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Self-Correction — the paper does NOT use few-shot demos for this
# stage; the prompt is purely the generic/gentle correction template. Exposed
# here for symmetry so callers can rely on a uniform per-stage API.
# ─────────────────────────────────────────────────────────────────────────────

def render_self_correction_demos() -> str:
    """DIN-SQL §4.4 does not use demonstrations for self-correction; returns
    an empty string. Kept as a separate entry-point so callers have a uniform
    `render_<stage>_demos()` interface across all four stages.
    """
    return ""


if __name__ == "__main__":
    # Smoke test — prints the first schema-linking demo so a reader can eyeball
    # the rendered format without running the full pipeline.
    block = render_schema_linking_demos(k=1)
    print("=== SCHEMA LINKING DEMO ===")
    print(block)
    print()
    print("=== CLASSIFICATION DEMO ===")
    print(render_classification_demos(k=1))
    print()
    print("=== SQL-GEN NON-NESTED DEMO ===")
    print(render_sql_gen_demos("NON-NESTED", k=1))
    print()
    print("=== SQL-GEN NESTED DEMO ===")
    print(render_sql_gen_demos("NESTED", k=1))
