"""Generate the social preview card and favicon for the write-up site.

`open-graph: true` alone gives a link preview with no image, which unfurls in Slack,
LinkedIn and X as a bare grey box. This renders the 1200x630 card those platforms
expect, plus a favicon.

    python scripts/make_social_card.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"


def social_card() -> None:
    mpl.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.06, 0.85, " ".join("CORRECTNESS-AWARE\u2009NL-TO-SQL\u2009OVER\u2009OCEL"),
             fontsize=13, color=BLUE, fontweight="700")
    fig.text(0.06, 0.705, "The model never", fontsize=45, color=INK, fontweight="700")
    fig.text(0.06, 0.585, "writes the SQL.", fontsize=45, color=INK, fontweight="700")
    fig.text(0.06, 0.485,
             "It proposes a typed plan. Deterministic checks decide what runs.",
             fontsize=17, color=MUTED)

    tiles = [
        ("Denotation accuracy", "64.9%", "vs 41.9% best baseline", BLUE),
        ("Relation-path errors", "0.0%", "held at zero", GREEN),
        ("Avg latency", "3.3s", "vs 10.0s DIN-SQL", ORANGE),
    ]
    for i, (label, value, sub, colour) in enumerate(tiles):
        x = 0.06 + i * 0.305
        fig.patches.append(
            mpl.patches.Rectangle((x, 0.11), 0.006, 0.20, transform=fig.transFigure,
                                  facecolor=colour, edgecolor="none")
        )
        fig.text(x + 0.022, 0.265, label, fontsize=13.5, color=MUTED)
        fig.text(x + 0.022, 0.165, value, fontsize=36, color=INK, fontweight="700")
        fig.text(x + 0.022, 0.122, sub, fontsize=12, color=colour, fontweight="600")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "social-card.png", facecolor=SURFACE, dpi=100)
    plt.close(fig)
    print("wrote assets/social-card.png (1200x630)")


def favicon() -> None:
    """A checkmark-through-brackets mark: verification gating a query."""
    fig = plt.figure(figsize=(0.64, 0.64), dpi=100)
    fig.patch.set_facecolor(BLUE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BLUE)
    ax.plot([0.26, 0.44, 0.76], [0.52, 0.32, 0.70], color="#ffffff",
            linewidth=7, solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(OUT / "favicon.png", facecolor=BLUE, dpi=100)
    plt.close(fig)
    print("wrote assets/favicon.png (64x64)")


if __name__ == "__main__":
    social_card()
    favicon()
