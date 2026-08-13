"""
nan_sampler.py — Nan et al. 2023 Similarity-Diversity demonstration sampling.

Implements the core prompt-design pipeline from:

    Linyong Nan, Yilun Zhao, Weijin Zou, Narutatsu Ri, Jaesung Tae,
    Ellen Zhang, Arman Cohan, Dragomir Radev.
    "Enhancing Few-shot Text-to-SQL Capabilities of Large Language Models:
     A Study on Prompt Design Strategies."
    Findings of EMNLP 2023.  arXiv:2305.12586.

The paper's key insight (§2.1): instead of retrieving demonstrations by the
**natural-language** semantic similarity of the question, retrieve by the
**SQL syntactic** structure of a draft query predicted for the test input.
This is called "Structured Prediction as Basis for Retrieval".

Full Algorithm 1 (Similarity-Diversity) — what this module builds:

    Input:  annotated pool A, test set T, number of demos k,
            categorization {alpha, beta, ...}  (we use Spider difficulty)
    Output: prompts P, one per test example

    1.  For each annotated example a in A:
            c_i = get_category(a.SQL)        # difficulty bucket
            A^{c_i}.append(a)
            V_i = syntax_vector(a.SQL)       # binary feature vector
    2.  For each partition A^j (difficulty bucket):
            M_j = k-Means(V^j, k)            # centroids in syntax space
            D^j = one annotated example nearest each centroid
    3.  For each test example t:
            t.SQL   = initial_predictor(t)   # cheap zero-shot draft
            c_t     = get_category(t.SQL)
            P_t     = build_prompt(D^{c_t}, t)

This module implements steps 1-3 independently of the LLM call. The caller
(`BaselineTranslator`) owns the initial predictor and prompt building.

Notes on fidelity
-----------------
- Spider difficulty taxonomy is defined in Yu et al. 2018. Spider has the
  labels "easy / medium / hard / extra-hard"; our OCEL benchmark uses
  "easy / medium / hard". We treat them as equivalent partitions.
- The paper's syntax vector uses SQL keywords + operators + identifiers.
  Identifiers (specific column/table names) would make vectors too sparse
  for a small pool; we follow the paper's public reference implementation
  (https://github.com/linyongnan/STRIKE) and restrict to keywords + ops.
- We use sqlglot for robust tokenization; fallback is regex for SQL that
  fails to parse.
"""

from __future__ import annotations

import numpy as np
import sqlglot
from sqlglot import exp


# ─────────────────────────────────────────────────────────────────────────────
# Feature set — the 28 binary features Nan 2023 uses to describe SQL syntax.
# Ordered so vector positions are stable across runs. Names match the upper-
# case tokens produced by sqlglot so we can detect them via AST node types
# or raw-token membership.
# ─────────────────────────────────────────────────────────────────────────────

_SYNTAX_FEATURES: list[str] = [
    # Core clauses
    "SELECT", "DISTINCT", "WHERE", "GROUP_BY", "HAVING",
    "ORDER_BY", "LIMIT", "WITH",               # WITH = CTE presence
    # Join styles
    "JOIN", "LEFT_JOIN", "INNER_JOIN", "CROSS_JOIN",
    # Set operators
    "UNION", "INTERSECT", "EXCEPT",
    # Aggregates
    "COUNT", "SUM", "AVG", "MAX", "MIN",
    # Predicate operators
    "IN", "NOT_IN", "EXISTS", "BETWEEN", "LIKE",
    "IS_NULL", "IS_NOT_NULL",
    # Date helpers (OCEL-specific but universally useful for text-to-SQL)
    "YEAR_FN", "MONTH_FN", "DATE_DIFF_FN",
]


def _empty_vector() -> np.ndarray:
    """Zero vector of the right length — used when SQL fails to tokenize."""
    return np.zeros(len(_SYNTAX_FEATURES), dtype=np.int8)


def syntax_vector(sql: str) -> np.ndarray:
    """Return a 0/1 vector indicating which of the Nan 2023 syntax features
    occur anywhere in the given SQL.

    Uses a lowercase-normalized token scan instead of a full AST walk. This
    is robust to dialect quirks (e.g. DuckDB's DATE_DIFF vs SQLite's
    strftime) and matches the simplicity of Nan's published STRIKE code.

    Parameters
    ----------
    sql : str
        A SELECT / WITH statement.

    Returns
    -------
    np.ndarray  (shape=(28,), dtype=int8)
        Binary presence vector. Order matches `_SYNTAX_FEATURES`.
    """
    if not sql or not sql.strip():
        return _empty_vector()

    # Optional sqlglot parse — if it succeeds we can be more accurate on a
    # few ambiguous cases (DISTINCT inside COUNT, INNER JOIN keyword).
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        parsed = None

    # Lowercase the SQL for case-insensitive token matching; collapse any
    # whitespace so we can search for multi-word keywords like "GROUP BY".
    text = " ".join(sql.lower().split())

    import re as _re

    def has(token: str) -> bool:
        """Token presence test.

        Behavior:
          - Multi-word tokens ('group by'): plain substring match on normalized text.
          - Tokens ending in '(' (e.g. 'count('): plain substring match — the '('
            itself gives the boundary, so "count(*)" correctly sets COUNT.
          - Single-word tokens ('where'): word-boundary regex so "where" does
            not also match the "where" inside some longer identifier.
        """
        if " " in token or token.endswith("("):
            return token in text
        return bool(_re.search(rf"\b{_re.escape(token)}\b", text))

    v = _empty_vector()

    def set_feat(name: str, present: bool) -> None:
        """Set feature bit by name."""
        if present:
            v[_SYNTAX_FEATURES.index(name)] = 1

    # Clauses.
    set_feat("SELECT",   has("select"))
    set_feat("DISTINCT", has("distinct"))
    set_feat("WHERE",    has("where"))
    set_feat("GROUP_BY", has("group by"))
    set_feat("HAVING",   has("having"))
    set_feat("ORDER_BY", has("order by"))
    set_feat("LIMIT",    has("limit"))
    set_feat("WITH",     has("with") and parsed is not None and isinstance(parsed, (exp.Select, exp.With)) or
                                    text.startswith("with "))

    # Join styles.
    set_feat("LEFT_JOIN",  has("left join"))
    set_feat("INNER_JOIN", has("inner join"))
    set_feat("CROSS_JOIN", has("cross join"))
    set_feat("JOIN",       has("join"))  # generic — includes the above

    # Set operators.
    set_feat("UNION",     has("union"))
    set_feat("INTERSECT", has("intersect"))
    set_feat("EXCEPT",    has("except"))

    # Aggregates.
    set_feat("COUNT", has("count("))
    set_feat("SUM",   has("sum("))
    set_feat("AVG",   has("avg("))
    set_feat("MAX",   has("max("))
    set_feat("MIN",   has("min("))

    # Predicate operators.
    set_feat("IN",        has(" in ") or has(" in("))
    set_feat("NOT_IN",    has("not in"))
    set_feat("EXISTS",    has("exists"))
    set_feat("BETWEEN",   has("between"))
    set_feat("LIKE",      has("like"))
    set_feat("IS_NULL",   has("is null"))
    set_feat("IS_NOT_NULL", has("is not null"))

    # Date helpers.
    set_feat("YEAR_FN",      has("year("))
    set_feat("MONTH_FN",     has("month("))
    set_feat("DATE_DIFF_FN", has("date_diff(") or has("datediff("))

    return v


def vector_feature_names() -> list[str]:
    """Expose the feature names for debugging / unit testing."""
    return list(_SYNTAX_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Spider-style difficulty classifier  (Yu et al. 2018, Appendix A)
#
# Nan 2023 §2.1 partitions the annotated pool by Spider difficulty
# ("easy / medium / hard / extra"). We reproduce the Spider hardness rule on
# SQL syntax here so that both pool-partitioning and test-question routing
# use the same, reproducible definition.
#
# The Spider definition counts the following "components" in a query:
#   SELECT items > 1, WHERE items > 1, GROUP BY > 1, ORDER BY > 1, LIMIT,
#   aggregate functions, set operators, nesting, JOIN.
# Thresholds (Yu 2018 Appendix A):
#   easy     : <= 1 component
#   medium   : 2 components OR a single JOIN
#   hard     : 3 components OR nesting / aggregation on groups
#   extra    : 4+ components
# Our benchmark uses three labels (easy / medium / hard); we map "extra" to
# "hard" since the OCEL benchmark does not distinguish them.
# ─────────────────────────────────────────────────────────────────────────────

def classify_difficulty(sql: str) -> str:
    """Return 'easy' | 'medium' | 'hard' for a given SQL string.

    Rough but principled implementation of the Spider hardness rule from
    Yu 2018 Appendix A. Uses the binary `syntax_vector` + a small count of
    JOIN / nesting tokens. Matches Nan 2023's pool-partitioning intent
    without re-implementing the full Spider parser.
    """
    if not sql or not sql.strip():
        return "easy"

    v = syntax_vector(sql)
    idx = {name: i for i, name in enumerate(_SYNTAX_FEATURES)}

    # Count the "components" that Spider considers hardness-contributing.
    components = 0
    if v[idx["GROUP_BY"]]:   components += 1
    if v[idx["HAVING"]]:     components += 1
    if v[idx["ORDER_BY"]]:   components += 1
    if v[idx["LIMIT"]]:      components += 1
    if v[idx["UNION"]] or v[idx["INTERSECT"]] or v[idx["EXCEPT"]]: components += 2  # set ops are heavy
    if v[idx["WITH"]]:       components += 2   # CTE counts as nesting
    if v[idx["EXISTS"]] or v[idx["NOT_IN"]]:   components += 1
    agg = any(v[idx[a]] for a in ("COUNT", "SUM", "AVG", "MAX", "MIN"))
    if agg: components += 1
    has_join = bool(v[idx["JOIN"]])

    # Nested sub-query detection: count extra "(select" occurrences.
    nested_selects = sql.lower().count("(select")
    if nested_selects >= 1: components += 2

    if components >= 3:
        return "hard"
    if components == 2 or has_join:
        return "medium"
    return "easy"


# ─────────────────────────────────────────────────────────────────────────────
# Similarity-Diversity sampler  (Nan 2023 Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityDiversitySampler:
    """
    Build a per-difficulty pool of annotated examples and, for a given test
    question's draft SQL, return k demonstrations chosen by the Nan 2023
    Similarity-Diversity algorithm:

      1.  Partition the annotated pool by Spider difficulty.
      2.  In each partition, run k-Means over the 28-dim syntax vectors and
          pick the single example nearest each centroid. These are the
          "Diverse" examples for that partition (Algorithm 1 step "for mu_i
          in M: D^j.append(get_nearest(A, mu_i))").
      3.  At query time, route to the partition matching the test's draft-SQL
          difficulty, and return those k diverse examples.

    This module handles steps 1-3; the caller owns the draft-SQL production.
    """

    def __init__(
        self,
        pool_rows: list[dict],
        k: int = 5,
        random_state: int = 42,
    ):
        """Build per-difficulty partitions and cluster them.

        Parameters
        ----------
        pool_rows : list of dict
            Annotated examples. Each dict must have keys `qid`, `nl_question`,
            `gold_sql`. An optional `difficulty` key overrides the computed
            Spider label; otherwise `classify_difficulty` is used.
        k : int
            Number of demonstrations to retrieve per query (matches the
            paper's 4..10 shot sweep). We pre-cluster to `k` centroids; the
            runtime caller may request fewer via `select(k=...)`.
        random_state : int
            Reproducibility seed for sklearn KMeans.
        """
        from sklearn.cluster import KMeans

        self.k = int(k)
        self._random_state = random_state
        # Compute (vector, row, difficulty) triples for every pool row.
        enriched: list[tuple[np.ndarray, dict, str]] = []
        for row in pool_rows:
            diff = (row.get("difficulty") or classify_difficulty(row["gold_sql"])).lower()
            if diff == "extra":
                diff = "hard"
            enriched.append((syntax_vector(row["gold_sql"]), row, diff))

        # Bucket by difficulty so step 1 of Algorithm 1 is done once.
        self._partitions: dict[str, list[tuple[np.ndarray, dict]]] = {
            "easy": [], "medium": [], "hard": [],
        }
        for vec, row, diff in enriched:
            self._partitions.setdefault(diff, []).append((vec, row))

        # Pre-cluster each partition so runtime `select()` is O(k).
        # If a partition has fewer than k points we fall back to "return all"
        # which matches Nan's implementation when the pool is small.
        self._per_partition_demos: dict[str, list[dict]] = {}
        for diff, pairs in self._partitions.items():
            if not pairs:
                self._per_partition_demos[diff] = []
                continue
            vectors = np.stack([vec for vec, _ in pairs])
            n_clusters = min(self.k, len(pairs))
            if n_clusters == len(pairs):
                # Partition too small to cluster — return everything once.
                self._per_partition_demos[diff] = [row for _, row in pairs]
                continue
            km = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=random_state,
            )
            km.fit(vectors)
            selected_rows: list[dict] = []
            used_indices: set[int] = set()
            for centroid in km.cluster_centers_:
                # pick nearest pool row not already picked (argmin w/ dedup)
                distances = np.linalg.norm(vectors - centroid, axis=1)
                order = np.argsort(distances)
                for idx in order:
                    if idx not in used_indices:
                        used_indices.add(int(idx))
                        selected_rows.append(pairs[int(idx)][1])
                        break
            self._per_partition_demos[diff] = selected_rows

    def select(
        self,
        draft_sql: str,
        k: int | None = None,
        exclude_qid: str | None = None,
    ) -> list[dict]:
        """Return k demonstrations matching the draft-SQL's difficulty.

        Parameters
        ----------
        draft_sql : str
            The initial predictor's SQL (produced by a cheap zero-shot call).
            Used only to route to a partition — the demonstrations themselves
            are pre-selected at construction time.
        k : int, optional
            Override the class-level k. When fewer than k demos are available
            for the routed partition we fall back to the full pool.
        exclude_qid : str, optional
            If provided, drop any demonstration row whose `qid` matches this
            value. This is the dev-set leave-one-out guard that prevents the
            current test question from appearing in its own demonstration
            pool when the same dev split is used as both eval target and
            sampler pool.
        """
        k_eff = k if k is not None else self.k
        diff = classify_difficulty(draft_sql)
        demos = list(self._per_partition_demos.get(diff, []))
        if exclude_qid is not None:
            demos = [d for d in demos if d.get("qid") != exclude_qid]
        if len(demos) < k_eff:
            # Partition thin: top up from other partitions (still deterministic
            # because we iterate dict in insertion order).
            topup: list[dict] = []
            for other_diff, pool in self._per_partition_demos.items():
                if other_diff == diff:
                    continue
                topup.extend(pool)
                if len(demos) + len(topup) >= k_eff:
                    break
            if exclude_qid is not None:
                topup = [d for d in topup if d.get("qid") != exclude_qid]
            demos = demos + topup[: max(0, k_eff - len(demos))]
        return demos[:k_eff]


# ─────────────────────────────────────────────────────────────────────────────
# Schema augmentation (Nan 2023 §2.2)
#
# The paper proposes TWO forms of schema augmentation, both added as comments
# on / near CREATE TABLE statements:
#   - Semantic augmentation   : one-line natural-language summary per column
#   - Structure augmentation  : a mini Entity-Relationship summary listing
#                               FK paths between tables in descending length
# We implement both below so the few-shot prompt can be the paper's best
# configuration "SD + SA + Voting" (Figure 1b).
# ─────────────────────────────────────────────────────────────────────────────

# Hand-written one-line summaries for every OCEL column. These are the exact
# semantic augmentations the few-shot prompt will embed as block comments
# inside each CREATE TABLE. Keeping them as a const dict — NOT generating at
# runtime — ensures the prompt is identical across runs (reproducibility).
_COLUMN_SUMMARIES: dict[tuple[str, str], str] = {
    # events
    ("events", "event_id"):     "unique identifier of the event record",
    ("events", "event_type"):   "business activity name such as billing_created, order_created, payment_clearing",
    ("events", "timestamp"):    "wall-clock time at which the business activity took place",
    ("events", "object_id"):    "identifier of the OCEL object the event refers to",
    ("events", "object_type"):  "OCEL object category (order_item, billing_doc, customer, ...)",
    ("events", "source_table"): "SAP source table the event was extracted from (VBAK, VBRK, BSAD, ...)",
    # objects
    ("objects", "object_id"):   "unique object identifier; joinable with events.object_id",
    ("objects", "object_type"): "OCEL object category (same vocabulary as events.object_type)",
    ("objects", "FKDAT"):       "billing date (billing_doc objects only)",
    ("objects", "NETWR"):       "net amount in document currency (billing_doc only)",
    ("objects", "WAERK"):       "document currency ISO code (billing_doc only)",
    ("objects", "ZTERM"):       "payment terms key (billing_doc only)",
    ("objects", "KUNAG"):       "sold-to customer number (billing_doc only)",
    ("objects", "BUKRS"):       "company code (ar_item only)",
    ("objects", "AUGDT"):       "clearing date (ar_item only) — when payment cleared",
    ("objects", "WRBTR"):       "amount in local currency (ar_item only)",
    ("objects", "WAERS"):       "local currency ISO code (ar_item only)",
    ("objects", "KUNNR"):       "payer customer number (ar_item only)",
    ("objects", "XBLNR"):       "reference document number (ar_item only)",
    ("objects", "cleared"):     "'X' if the AR item has been cleared, NULL otherwise",
    # relations
    ("relations", "from_object_id"):   "origin side of the object-to-object link",
    ("relations", "from_object_type"): "OCEL type of the origin object",
    ("relations", "to_object_id"):     "destination side of the object-to-object link",
    ("relations", "to_object_type"):   "OCEL type of the destination object",
    ("relations", "relation_type"):    "name of the allowed link (order_to_delivery, delivery_to_billing, ...)",
    ("relations", "source_table"):     "SAP source table the relation was extracted from",
}


def build_semantic_augmentation_comments(catalog: dict) -> dict[str, str]:
    """For each table, return a multi-line SQL comment block that lists
    every column and its one-line semantic description.

    Matches Nan 2023 §2.2 "Semantic augmentation — add column summary
    (block-comment)". The block is injected next to each CREATE TABLE in
    the final prompt.

    Output dict keyed by table_name.
    """
    out: dict[str, str] = {}
    for tbl in catalog["tables"]:
        name = tbl["table_name"]
        lines = ["/*", f"Column descriptions for {name}:"]
        for col in tbl["columns"]:
            summary = _COLUMN_SUMMARIES.get((name, col["name"]),
                                            "(no description available)")
            lines.append(f"  - {col['name']}: {summary}")
        lines.append("*/")
        out[name] = "\n".join(lines)
    return out


def build_structure_augmentation_comment(whitelist: list[dict]) -> str:
    """Entity-Relationship summary block (Nan 2023 §2.2 "Structure
    augmentation — add ontology summary").

    Enumerates every allowed FK path across tables; sorted by path length
    so that multi-hop paths appear first, matching Figure 9 of the Nan
    appendix (which shows "paths arranged in descending length order").
    """
    # Direct 1-hop paths from the whitelist.
    one_hop = [
        (w["from_object_type"], w["to_object_type"], w["relation_type"])
        for w in whitelist
    ]

    # Derive 2-hop paths by chaining compatible edges (from_type of one
    # edge == to_type of another). Only compose distinct relation types.
    two_hop: list[tuple[str, str, str, str]] = []
    for a_from, a_to, a_rel in one_hop:
        for b_from, b_to, b_rel in one_hop:
            if a_to == b_from and a_rel != b_rel:
                two_hop.append((a_from, a_to, b_to, f"{a_rel} -> {b_rel}"))

    lines = ["/*", "Entity-Relationship summary (longest paths first):"]
    for src, mid, dst, chain in two_hop:
        lines.append(f"  {src} -> {mid} -> {dst}   via  {chain}")
    for src, dst, rel in one_hop:
        lines.append(f"  {src} -> {dst}            via  {rel}")
    lines.append("*/")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NanFewShotPipeline  —  orchestrates the full Algorithm 2 "Integrated Strategy"
# ─────────────────────────────────────────────────────────────────────────────

class NanFewShotPipeline:
    """Full Nan 2023 few-shot pipeline for one backend + one OCEL schema.

    Responsibilities
    ----------------
    1. Hold the annotated pool + pre-built SimilarityDiversitySampler.
    2. Drive the initial_predictor (a caller-supplied callable returning
       a draft SQL for a natural-language question).
    3. Build the code-sequence CREATE TABLE schema + semantic augmentation
       + structure augmentation (Nan §2.2 "code-seq + SA").
    4. Assemble the final prompt and run the integrated voting strategy
       (Algorithm 2).

    The class is LLM-agnostic: both the `initial_predictor` and the
    `final_predictor` are injected as callables `str -> str`. An `executor`
    callable (`str -> tuple[ok: bool, result_hash: str]`) is used by the
    voting loop to discard exec-failing predictions.
    """

    def __init__(
        self,
        pool_rows: list[dict],
        catalog: dict,
        whitelist: list[dict],
        k: int = 10,
        random_state: int = 42,
    ):
        self.sampler = SimilarityDiversitySampler(
            pool_rows=pool_rows, k=k, random_state=random_state,
        )
        self._catalog   = catalog
        self._whitelist = whitelist
        self._semantic_comments  = build_semantic_augmentation_comments(catalog)
        self._structure_comment  = build_structure_augmentation_comment(whitelist)

    # -- Step 2.2 "code-seq" schema (Nan used CREATE TABLE style) -------------

    def _build_code_seq_schema(self) -> str:
        """Render the full schema as CREATE TABLE blocks interleaved with
        per-table semantic-augmentation comments and a final structure-
        augmentation comment. Identical across all questions — the prompt
        varies only in the demonstrations and the test question tail.
        """
        # Reuse the Rajkumar CREATE TABLE builder from baseline_translator
        # (it already produces clean CREATE TABLE statements). Doing the
        # import here rather than at module top avoids a cycle.
        from .baseline_translator import _build_create_table_block

        blocks: list[str] = []
        for tbl in self._catalog["tables"]:
            blocks.append(_build_create_table_block(tbl))
            sem = self._semantic_comments.get(tbl["table_name"])
            if sem:
                blocks.append(sem)
        blocks.append(self._structure_comment)
        return "\n\n".join(blocks)

    # -- Step 2.3 "Integrated Strategy" prompt body ---------------------------

    def _render_demos(self, demos: list[dict]) -> str:
        """Format demonstrations in the `<Q, SQL>` layout Nan 2023 Figure 11
        uses. One demo per block, separated by the delimiter `-- ###`.
        """
        blocks: list[str] = []
        for d in demos:
            blocks.append(
                f"-- Question: {d['nl_question']}\n"
                f"{d['gold_sql'].rstrip(';')};"
            )
        return "\n\n-- ###\n\n".join(blocks)

    def build_prompt(
        self,
        question: str,
        draft_sql: str,
        n_shots: int,
        exclude_qid: str | None = None,
    ) -> str:
        """Assemble the final few-shot prompt for one (question, n_shots) combo.

        Matches Nan 2023 Figure 11 layout:

            <CREATE TABLE events (...); + semantic comment>
            <CREATE TABLE objects (...); + semantic comment>
            <CREATE TABLE relations (...); + semantic comment>
            <structure-augmentation ER summary>

            <demo 1>
            -- ###
            <demo 2>
            ...

            -- Question: <test question>
        """
        demos = self.sampler.select(draft_sql, k=n_shots, exclude_qid=exclude_qid)
        demo_block = self._render_demos(demos)
        schema_block = self._build_code_seq_schema()
        return (
            f"{schema_block}\n\n"
            f"{demo_block}\n\n"
            f"-- Question: {question}\n"
        )

    # -- Step "Integrated Strategy" voting loop  (Algorithm 2) ----------------

    def predict_with_voting(
        self,
        question: str,
        initial_predictor,                   # callable: str -> str
        final_predictor,                     # callable: str -> dict with 'text' or str
        executor,                            # callable: str -> tuple[bool, str] (ok, result_hash)
        shot_range: range | list[int] = range(4, 11),
        exclude_qid: str | None = None,      # leave-one-out guard for dev eval
    ) -> dict:
        """Run the full Nan 2023 Integrated Strategy (Algorithm 2) for one test question.

        Parameters
        ----------
        question : str
            Natural-language question (test input).
        initial_predictor : callable (str -> str)
            Takes a natural-language question and returns a draft SQL string.
            Used only to route the test to a difficulty partition. Typical
            choice: the zero-shot B1 translator (Rajkumar 2022 "Create Table
            + Select 3" prompt, so the draft already contains schema links).
        final_predictor : callable (str -> str | dict)
            Takes the full few-shot prompt and returns the candidate SQL.
            May return a plain string OR a dict with keys `text` / `error` /
            `latency_s`; both forms are accepted for convenience.
        executor : callable (str -> tuple[bool, str])
            Runs the candidate SQL against the live OCEL DuckDB database and
            returns `(exec_ok, result_hash)`. A hash of "" / None means the
            query crashed or returned nothing.
        shot_range : iterable of int
            Shot counts to sweep. Paper default = 4..10 inclusive.

        Returns
        -------
        dict
            {
              'pred_sql'       : str — the majority-voted final SQL,
              'vote_hash'      : str — the winning result_hash,
              'draft_sql'      : str — the zero-shot draft used for routing,
              'difficulty'     : str — routed partition ('easy'|'medium'|'hard'),
              'per_shot'       : list[dict] — one entry per shot count:
                                 {'n_shots', 'pred_sql', 'exec_ok',
                                  'result_hash', 'latency_s'},
              'total_llm_calls': int   — 1 (initial) + len(shot_range).
            }

        When every shot produces an exec error, the first draft is returned
        as a graceful fallback (matches Nan's reference implementation
        behaviour on the "everything crashed" edge case).
        """
        from collections import Counter
        import time

        # -- 1. Initial predictor -> draft SQL -------------------------------
        draft_sql = initial_predictor(question) or ""
        draft_sql = _coerce_sql(draft_sql)
        difficulty = classify_difficulty(draft_sql)

        # -- 2. Sweep shot counts --------------------------------------------
        per_shot: list[dict] = []
        shot_list = list(shot_range)
        for n_shots in shot_list:
            prompt = self.build_prompt(
                question, draft_sql=draft_sql, n_shots=n_shots,
                exclude_qid=exclude_qid,
            )
            t0 = time.time()
            raw = final_predictor(prompt)
            candidate_sql = _coerce_sql(raw)
            ok, result_hash = executor(candidate_sql) if candidate_sql else (False, "")
            per_shot.append({
                "n_shots":     n_shots,
                "pred_sql":    candidate_sql,
                "exec_ok":     bool(ok),
                "result_hash": result_hash or "",
                "latency_s":   round(time.time() - t0, 3),
            })

        # -- 3. Remove exec-error predictions, majority-vote on result hash --
        surviving = [p for p in per_shot if p["exec_ok"] and p["result_hash"]]
        if surviving:
            vote_counter = Counter(p["result_hash"] for p in surviving)
            vote_hash, _ = vote_counter.most_common(1)[0]
            # Among shots that produced the winning hash, prefer the one with
            # the highest shot count (paper's tie-break: "larger n tends to
            # be more reliable").
            winners = [p for p in surviving if p["result_hash"] == vote_hash]
            winners.sort(key=lambda p: p["n_shots"], reverse=True)
            final_sql = winners[0]["pred_sql"]
        else:
            # Every shot crashed — fall back to the draft SQL so downstream
            # evaluation still gets a string to execute / hash.
            vote_hash = ""
            final_sql = draft_sql

        return {
            "pred_sql":        final_sql,
            "vote_hash":       vote_hash,
            "draft_sql":       draft_sql,
            "difficulty":      difficulty,
            "per_shot":        per_shot,
            "total_llm_calls": 1 + len(shot_list),
        }


def _coerce_sql(raw) -> str:
    """Accept either a plain string or an `{text, error, latency_s}` dict from
    the LLM client and return the stripped SQL body.
    """
    if isinstance(raw, dict):
        if raw.get("error"):
            return ""
        text = raw.get("text", "")
    else:
        text = raw or ""
    # Reuse the same SQL extraction heuristic as baseline_translator.extract_sql.
    from .baseline_translator import extract_sql
    return extract_sql(text)


if __name__ == "__main__":
    # Smoke test over some hand-made SQL and a miniature "pool".
    tests = [
        "SELECT COUNT(*) FROM events",
        "SELECT object_type, COUNT(*) FROM events GROUP BY object_type HAVING COUNT(*) > 10",
        "WITH t AS (SELECT * FROM events WHERE YEAR(timestamp) = 2005) "
        "SELECT AVG(DATE_DIFF('day', a.ts, b.ts)) FROM t a JOIN t b ON a.object_id = b.object_id",
    ]
    names = vector_feature_names()
    for t in tests:
        v = syntax_vector(t)
        active = [n for n, bit in zip(names, v) if bit]
        diff = classify_difficulty(t)
        print(f"[{diff:<6}]", t[:80])
        print("  -> active features:", active)
    print()

    pool = [
        {"qid": "P01", "nl_question": "count", "gold_sql": "SELECT COUNT(*) FROM events"},
        {"qid": "P02", "nl_question": "year",  "gold_sql": "SELECT COUNT(*) FROM events WHERE YEAR(timestamp) = 2004"},
        {"qid": "P03", "nl_question": "group", "gold_sql": "SELECT object_type, COUNT(*) FROM events GROUP BY object_type"},
        {"qid": "P04", "nl_question": "join",  "gold_sql": "SELECT COUNT(DISTINCT from_object_id) FROM relations WHERE relation_type = 'order_to_delivery'"},
        {"qid": "P05", "nl_question": "nest",  "gold_sql": "WITH x AS (SELECT object_id FROM events) SELECT COUNT(*) FROM x"},
        {"qid": "P06", "nl_question": "join2", "gold_sql": "SELECT COUNT(*) FROM events e JOIN objects o ON e.object_id = o.object_id"},
    ]
    sampler = SimilarityDiversitySampler(pool, k=2)
    demo_ids = [r["qid"] for r in sampler.select("SELECT COUNT(*) FROM events WHERE YEAR(timestamp) = 2005", k=2)]
    print("Selected demos for an EASY draft:", demo_ids)
    demo_ids = [r["qid"] for r in sampler.select("WITH x AS (SELECT * FROM events) SELECT COUNT(*) FROM x", k=2)]
    print("Selected demos for a HARD draft:", demo_ids)
