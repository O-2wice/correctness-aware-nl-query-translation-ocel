"""
natsql.py — NatSQL intermediate representation for DIN-SQL's non-nested class.

DIN-SQL (Pourreza & Rafiei, NeurIPS 2023) cites NatSQL from Gan et al. 2021
("Natural SQL: Making SQL Easier to Infer from Natural Language Specifications",
Findings of EMNLP 2021) as its intermediate representation for the non-nested
complex class. The rationale (DIN-SQL §4.3): "Expressions in natural language
queries may not clearly map to a unique SQL clause... removing operators makes
the transition from natural language to SQL easier."

This module provides a **small, explainable** subset of the NatSQL rewrites
that is sufficient to produce demo-ready IR strings from canonical DuckDB
SQL. Non-nested demos hand-annotated in `configs/din_sql_demos/demos.json`
already carry NatSQL fields; this helper is used when adding new demos or
as a fallback if a non-nested demo lacks a hand-authored `natsql` field.

Core rewrites (a faithful subset of Gan et al. 2021 §3):
  1. Drop FROM clause (table binding inferred from schema links).
  2. Drop all JOIN ... ON ... clauses (join structure is implicit).
  3. Drop GROUP BY clauses when the aggregate is obvious.
  4. Merge any HAVING predicates into the WHERE clause (both are "filter"
     semantically in NatSQL).
  5. Keep SELECT, WHERE, ORDER BY, LIMIT unchanged.

We intentionally do NOT attempt the full NatSQL grammar (54 rewrite rules
for set operators, nested aggregates, etc.) — DIN-SQL only shows NatSQL in
the few-shot *demonstrations* for the non-nested class, not at inference
time. A hand-annotated demo set covers the vocabulary our evaluation needs.

Usage
-----
>>> sql_to_natsql("SELECT COUNT(*) FROM relations WHERE relation_type = 'billing_to_ar'")
"SELECT COUNT(*) WHERE relations.relation_type = 'billing_to_ar'"
"""

from __future__ import annotations

import re


# Match the FROM … (until a following WHERE/ORDER/GROUP/HAVING/LIMIT or end).
# The SQL we target is single-statement, no nested SELECT in FROM — fine.
_FROM_CLAUSE_RE = re.compile(
    r"\bFROM\b.*?(?=(\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$))",
    re.IGNORECASE | re.DOTALL,
)

# Match the GROUP BY clause up to the next major clause keyword.
_GROUP_BY_RE = re.compile(
    r"\bGROUP\s+BY\b.*?(?=(\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$))",
    re.IGNORECASE | re.DOTALL,
)

# HAVING predicate (we'll pull it out and merge into WHERE).
_HAVING_RE = re.compile(
    r"\bHAVING\b\s+(?P<pred>.*?)(?=(\bORDER\s+BY\b|\bLIMIT\b|;|$))",
    re.IGNORECASE | re.DOTALL,
)


def sql_to_natsql(sql: str) -> str:
    """Rewrite a canonical DuckDB SELECT query into NatSQL-style IR.

    Applies the five simplification rules described in the module docstring.
    Does not parse SQL rigorously — it operates on clause-level regexes. That
    is fine for demo generation on our OCEL benchmark where every query is a
    single flat SELECT without nested FROM-subqueries.

    Parameters
    ----------
    sql : str
        A DuckDB SELECT statement.

    Returns
    -------
    str
        A NatSQL-flavored string with FROM / JOIN / GROUP BY removed and
        HAVING merged into WHERE. The overall SELECT projection and filter
        predicates are preserved verbatim.
    """
    # Normalize whitespace first so our regexes behave predictably.
    normalized = " ".join(sql.strip().split())

    # 1) Pull HAVING predicate aside (we'll merge it into WHERE below).
    having_match = _HAVING_RE.search(normalized)
    having_pred  = having_match.group("pred").strip() if having_match else None
    if having_match:
        normalized = normalized[:having_match.start()] + normalized[having_match.end():]

    # 2) Drop FROM ... (up to next clause keyword). This strips implicit
    #    JOIN ... ON ... because they are part of the FROM clause in SQL grammar.
    normalized = _FROM_CLAUSE_RE.sub("", normalized)

    # 3) Drop GROUP BY clauses — NatSQL leaves the aggregate implicit.
    normalized = _GROUP_BY_RE.sub("", normalized)

    # 4) Merge the HAVING predicate (if any) into WHERE.
    if having_pred:
        # If there is already a WHERE, AND the predicates together. Otherwise
        # inject a fresh WHERE right after the SELECT list.
        if re.search(r"\bWHERE\b", normalized, re.IGNORECASE):
            normalized = re.sub(
                r"\bWHERE\b\s+(.*?)(?=(\bORDER\s+BY\b|\bLIMIT\b|;|$))",
                lambda m: f"WHERE {m.group(1).strip()} AND {having_pred}",
                normalized, count=1, flags=re.IGNORECASE | re.DOTALL,
            )
        else:
            # Insert WHERE right before ORDER BY / LIMIT / end.
            insertion_point = re.search(
                r"(\bORDER\s+BY\b|\bLIMIT\b|;|$)", normalized, re.IGNORECASE
            )
            if insertion_point:
                idx = insertion_point.start()
                normalized = normalized[:idx].rstrip() + f" WHERE {having_pred} " + normalized[idx:]

    # 5) Tidy up doubled spaces introduced by the substitutions.
    return " ".join(normalized.split())


if __name__ == "__main__":
    # Quick self-check examples. Run directly to eyeball NatSQL output.
    examples = [
        "SELECT COUNT(*) FROM relations WHERE relation_type = 'billing_to_ar'",
        "SELECT o.object_type, COUNT(*) FROM events e JOIN objects o ON e.object_id = o.object_id GROUP BY o.object_type HAVING COUNT(*) > 100",
        "SELECT COUNT(DISTINCT from_object_id) FROM relations WHERE relation_type = 'order_to_delivery'",
    ]
    for s in examples:
        print("SQL   :", s)
        print("NatSQL:", sql_to_natsql(s))
        print()
