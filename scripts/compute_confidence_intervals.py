"""
compute_confidence_intervals.py

Bootstrap 95% confidence intervals for ExecRate and DenAcc on dev and test
evaluation CSVs. Also runs McNemar's test between pairs of methods.

Usage:
    python scripts/compute_confidence_intervals.py

Output:
    Table of mean ± 95% CI for each method × metric × split.
    McNemar p-values for method pairs on DenAcc.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
REPORTS  = ROOT / "outputs" / "reports"
N_BOOT   = 10_000
RNG_SEED = 42

# After the Phase 0 labeling fix, each baseline writes to the CSV whose
# number matches its manuscript label: b1 → nb03_b1, b2 → nb03_b2, b3 → nb03_b3.
RESULT_FILES = {
    "B1 Zero-shot": {
        "dev":  REPORTS / "nb03_b1_results.csv",
        "test": REPORTS / "nb03_b1_results_test.csv",
    },
    "B2 Few-shot": {
        "dev":  REPORTS / "nb03_b2_results.csv",
        "test": REPORTS / "nb03_b2_results_test.csv",
    },
    "B3 DIN-SQL": {
        "dev":  REPORTS / "nb03_b3_results.csv",
        "test": REPORTS / "nb03_b3_results_test.csv",
    },
    "Method M": {
        "dev":  REPORTS / "nb04_method_m_dev.csv",
        "test": REPORTS / "nb04_method_m_test.csv",
    },
}


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple:
    """Return (mean, lower_95, upper_95) via percentile bootstrap."""
    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lo), float(hi)


def mcnemar_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """McNemar's test p-value (exact binomial, continuity corrected)."""
    from scipy.stats import binom  # type: ignore
    b01 = int(((~a_correct) & b_correct).sum())   # A wrong, B right
    b10 = int((a_correct & (~b_correct)).sum())    # A right, B wrong
    n   = b01 + b10
    if n == 0:
        return 1.0
    k   = min(b01, b10)
    # two-sided exact binomial: P(X ≤ k) * 2, capped at 1
    return min(2 * binom.cdf(k, n, 0.5), 1.0)


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # keep last run per qid if file has duplicates
    if "qid" in df.columns:
        df = df.drop_duplicates(subset="qid", keep="last").reset_index(drop=True)
    return df


def main():
    rows = []
    method_split_correct: dict[tuple, np.ndarray] = {}

    for method, splits in RESULT_FILES.items():
        for split, path in splits.items():
            df = load(path)
            if df is None:
                print(f"  [MISSING] {path.name} — skipping")
                continue

            exec_vals = df["exec_ok"].astype(float).values if "exec_ok" in df.columns \
                        else (df["status"] == "accept").astype(float).values
            den_vals  = df["den_acc"].astype(float).values

            exec_mean, exec_lo, exec_hi = bootstrap_ci(exec_vals)
            den_mean,  den_lo,  den_hi  = bootstrap_ci(den_vals)

            rows.append({
                "Method": method, "Split": split, "n": len(df),
                "ExecRate": f"{exec_mean:.1%} [{exec_lo:.1%}, {exec_hi:.1%}]",
                "DenAcc":   f"{den_mean:.1%} [{den_lo:.1%}, {den_hi:.1%}]",
                "Latency":  f"{df['latency_s'].mean():.1f}s",
            })
            method_split_correct[(method, split)] = den_vals.astype(bool)

    results = pd.DataFrame(rows)
    print("\n=== Bootstrap 95% CI (N_BOOT=10,000) ===\n")
    print(results.to_string(index=False))

    # ── McNemar tests: Method M vs each baseline ──────────────────────────────
    print("\n=== McNemar's test — Method M vs baselines (DenAcc) ===\n")
    for split in ("dev", "test"):
        m_correct = method_split_correct.get(("Method M", split))
        if m_correct is None:
            continue
        print(f"Split: {split}")
        for baseline in ("B1 Zero-shot", "B2 Few-shot", "B3 DIN-SQL"):
            b_correct = method_split_correct.get((baseline, split))
            if b_correct is None:
                continue
            if len(b_correct) != len(m_correct):
                print(f"  vs {baseline}: n mismatch, skip")
                continue
            p = mcnemar_p(m_correct, b_correct)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  Method M vs {baseline}: p={p:.4f} {sig}")
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out = REPORTS / "confidence_intervals.csv"
    results.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
