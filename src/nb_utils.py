"""
Shared notebook output formatting utilities.

Import in any notebook after ROOT is defined:
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from nb_utils import progress, styled_df, show_html
    from nb_utils import print_banner, print_rule, print_ok, print_warn
    from nb_utils import print_err, print_info, print_metric_table, print_gate

Requires: rich, tqdm, colorama, jinja2  (all in requirements.txt)
Falls back to plain text if any library is missing.
"""

import sys as _sys

_IN_JUPYTER = "ipykernel" in _sys.modules
_IN_COLAB = "google.colab" in _sys.modules

# ── Colab: auto-install missing packages ─────────────────────────────────
if _IN_COLAB:
    import subprocess as _sp

    for _pkg in ["rich", "colorama", "tqdm", "jinja2", "tabulate"]:
        try:
            __import__(_pkg)
        except ImportError:
            _sp.run([_sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

# ── Optional imports / capability flags ──────────────────────────────────
try:
    import colorama as _colorama

    _colorama.init(autoreset=True)
    _COLORAMA_OK = True
except ImportError:
    _COLORAMA_OK = False

try:
    from tqdm.auto import tqdm as _tqdm

    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False
    _tqdm = None

try:
    from IPython.display import HTML as _HTML
    from IPython.display import display as _display
    import pandas as _pd_fmt

    _IPYTHON_OK = True
except ImportError:
    _IPYTHON_OK = False
    _HTML = None
    _display = None
    _pd_fmt = None

try:
    from rich import box as _box
    from rich.console import Console as _Console
    from rich.panel import Panel as _Panel
    from rich.rule import Rule as _Rule
    from rich.table import Table as _Table

    _con = _Console(force_jupyter=_IN_JUPYTER, highlight=False)
    _RICH_OK = True
except ImportError:
    _RICH_OK = False
    _box = None
    _Console = None
    _Panel = None
    _Rule = None
    _Table = None
    _con = None


def progress(iterable, desc="", total=None):
    if _TQDM_OK:
        return _tqdm(
            iterable,
            desc=desc,
            total=total,
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=True,
        )
    return iterable


def styled_df(df, highlight_cols=None, cmap="RdYlGn", title=""):
    """Display a DataFrame as a clean HTML table. No index, no gradient."""
    if not _IPYTHON_OK or _HTML is None or _display is None:
        if title:
            print(title)
        print(df.to_string(index=False))
        return

    import math

    _non_numeric = {
        "table", "description", "model", "metric", "feature", "label",
        "indicator", "theme", "interpretation", "split", "horizon", "status",
        "group", "class", "category", "note", "item", "value", "name", "type",
        "key", "stage", "column", "field",
    }

    css = """<style>
      .nb-df-wrap {
        margin: 8px 0 16px 0;
        padding: 10px 12px;
        background: #ffffff;
        border: 1px solid #d8e0ec;
        border-radius: 6px;
        overflow-x: auto;
        color: #1f2937;
      }
      .nb-df-title {
        font-size: 14px;
        font-weight: 700;
        color: #102a43;
        margin-bottom: 8px;
      }
      table.nb-df { border-collapse: collapse; font-size: 12.5px; width: 100%;
                    font-family: 'Segoe UI', Arial, sans-serif; background: #ffffff; }
      table.nb-df thead tr { background: #eef2f8; }
      table.nb-df th { padding: 7px 14px; text-align: left; font-weight: 600;
                       color: #1a3a5c; border-bottom: 2px solid #c5d0e0;
                       white-space: nowrap; }
      table.nb-df th.r { text-align: right; }
      table.nb-df td { padding: 5px 14px; border-bottom: 1px solid #e8ecf2;
                       color: #222; vertical-align: top; line-height: 1.35; }
      table.nb-df td.r { text-align: right; font-variant-numeric: tabular-nums; }
      table.nb-df tbody tr { background: #ffffff; }
      table.nb-df tbody tr:nth-child(even) { background: #f8fafd; }
      table.nb-df tbody tr:hover { background: #edf2fb; }
    </style>"""

    def _right(col):
        return col.lower() not in _non_numeric

    title_html = f'<div class="nb-df-title">{title}</div>' if title else ""
    header = "".join(
        f'<th class="{"r" if _right(c) else ""}">{c}</th>' for c in df.columns
    )
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            try:
                s = "" if (isinstance(v, float) and math.isnan(v)) else str(v)
            except Exception:
                s = "" if v is None else str(v)
            cls = "r" if _right(c) else ""
            cells.append(f'<td class="{cls}">{s}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    html = (
        f'<div class="nb-df-wrap">{title_html}'
        f'<table class="nb-df"><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>'
    )
    _display(_HTML(css + html))


def show_html(html_str):
    if _IPYTHON_OK:
        _display(_HTML(html_str))
    else:
        print(html_str)


def print_banner(title, subtitle="", style="bold blue"):
    if _RICH_OK:
        body = f"[bold]{title}[/bold]"
        if subtitle:
            body += f"\n[dim]{subtitle}[/dim]"
        _con.print(_Rule(f"[bold]{title}[/bold]" + (f"  [dim]{subtitle}[/dim]" if subtitle else ""), style="blue"))
        return
    print("\n" + "=" * 65 + "\n  " + title + (f"\n  {subtitle}" if subtitle else "") + "\n" + "=" * 65)


def print_rule(title=""):
    if _RICH_OK:
        _con.print(_Rule(title, style="dim cyan"))
        return
    print("\n--- " + title + " " + "-" * max(0, 55 - len(title)))


def print_ok(msg):
    if _RICH_OK:
        _con.print(f"  [bold green]OK[/bold green]    {msg}")
    else:
        print("  OK    " + str(msg))


def print_warn(msg):
    if _RICH_OK:
        _con.print(f"  [bold yellow]WARN[/bold yellow]  {msg}")
    else:
        print("  WARN  " + str(msg))


def print_err(msg):
    if _RICH_OK:
        _con.print(f"  [bold red]ERR[/bold red]   {msg}")
    else:
        print("  ERR   " + str(msg))


def print_info(msg):
    if _RICH_OK:
        _con.print(f"  [bold cyan]INFO[/bold cyan]  {msg}")
    else:
        print("  INFO  " + str(msg))


def print_metric_table(rows, title="", columns=None, highlight_col=None):
    """Print a formatted results table. highlight_col marks the primary metric column."""
    if not rows:
        print_warn("No rows to display.")
        return

    cols = columns or list(rows[0].keys())

    non_numeric = {
        "model", "metric", "feature", "label", "indicator", "theme",
        "interpretation", "split", "horizon", "status", "group",
        "table", "description", "class", "category", "note", "item", "value",
    }

    if _IPYTHON_OK and _HTML is not None and _display is not None:
        css = """
        <style>
          .nb-metric-wrap { margin: 6px 0 10px 0; }
          .nb-metric-title {
            font-size: 13px; font-weight: 700; color: #1a3a5c;
            margin-bottom: 5px; letter-spacing: .01em;
          }
          table.nb-metric {
            border-collapse: collapse; font-size: 12px;
            width: 100%; font-family: 'Segoe UI', Arial, sans-serif;
          }
          table.nb-metric thead tr {
            background: #1a3a5c; color: #ffffff;
          }
          table.nb-metric th {
            padding: 7px 12px; text-align: left;
            font-weight: 600; letter-spacing: .02em;
            border-bottom: 2px solid #0d2137;
          }
          table.nb-metric th.nb-right { text-align: right; }
          table.nb-metric td {
            padding: 5px 12px; border-bottom: 1px solid #dde3ee;
            color: #1a1a2e;
          }
          table.nb-metric td.nb-right { text-align: right; font-variant-numeric: tabular-nums; }
          table.nb-metric td.nb-highlight { color: #1a6e2e; font-weight: 700; }
          table.nb-metric tbody tr:nth-child(even) { background: #f0f4fa; }
          table.nb-metric tbody tr:nth-child(odd)  { background: #ffffff; }
          table.nb-metric tbody tr:hover { background: #e4ecf7; }
        </style>
        """
        title_html = f'<div class="nb-metric-title">{title}</div>' if title else ""
        header_cells = "".join(
            f'<th class="{"nb-right" if c.lower() not in non_numeric else ""}">{c}</th>'
            for c in cols
        )
        body_rows = []
        for i, r in enumerate(rows):
            cells = []
            for c in cols:
                val = str(r.get(c, ""))
                right = c.lower() not in non_numeric
                hl = c == highlight_col
                cls = " ".join(filter(None, [
                    "nb-right" if right else "",
                    "nb-highlight" if hl else "",
                ]))
                cells.append(f'<td class="{cls}">{val}</td>')
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        html = (
            f'<div class="nb-metric-wrap">{title_html}'
            f'<table class="nb-metric"><thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table></div>'
        )
        _display(_HTML(css + html))
        return

    import pandas as _pd2
    if title:
        print("\n" + title)
    print(_pd2.DataFrame(rows).to_string(index=False))


def print_gate(name, passed, value="", threshold="", warn_only=False):
    """Print a quality gate result with coloured pass/fail icon."""
    if passed:
        icon_rich = "[bold green]PASS[/bold green]"
        icon_plain = "PASS"
    elif warn_only:
        icon_rich = "[bold yellow]WARN[/bold yellow]"
        icon_plain = "WARN"
    else:
        icon_rich = "[bold red]FAIL[/bold red]"
        icon_plain = "FAIL"

    if _RICH_OK:
        _con.print(f"  [{icon_rich}] {name}: {value}  (threshold: {threshold})")
    else:
        print(f"  [{icon_plain}] {name}: {value}  (threshold: {threshold})")


def get_project_root():
    """Resolve the project root from any notebook working directory.

    Checks (in order): O2C_PROJECT_ROOT env var, OCEL_NLQ_PROJECT_ROOT env var,
    cwd, cwd parent. Identifies root by presence of data/ and notebooks/.
    """
    import os
    from pathlib import Path as _Path
    for key in ("O2C_PROJECT_ROOT", "OCEL_NLQ_PROJECT_ROOT"):
        v = os.environ.get(key)
        if v and _Path(v).exists():
            return _Path(v).resolve()
    for cand in [_Path.cwd(), _Path.cwd().parent]:
        cand = cand.resolve()
        if (cand / "data").exists() and (cand / "notebooks").exists():
            return cand
    raise RuntimeError(
        "Cannot locate project root. Set O2C_PROJECT_ROOT env var or run from "
        "the notebooks/ or project root directory."
    )


def save_fig(name: str, figures_dir, dpi: int = 180):
    """Save the current matplotlib figure as PDF and PNG, then show it.

    Replaces the repeated pattern:
        plt.savefig(FIGURES / 'name.pdf')
        plt.savefig(FIGURES / 'name.png', dpi=180)
        plt.show()
    with a single call: save_fig('name', FIGURES)
    """
    try:
        import matplotlib.pyplot as _plt
        from pathlib import Path as _Path
        d = _Path(figures_dir)
        _plt.savefig(d / f"{name}.pdf", bbox_inches="tight")
        _plt.savefig(d / f"{name}.png", dpi=dpi, bbox_inches="tight")
        _plt.show()
    except Exception:
        pass


def setup_matplotlib():
    """Apply project-wide matplotlib style. Call once per notebook session."""
    try:
        import matplotlib as _mpl

        _mpl.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.dpi": 150,
                "savefig.dpi": 180,
                "savefig.bbox": "tight",
            }
        )
    except ImportError:
        pass


setup_matplotlib()

__all__ = [
    "progress",
    "styled_df",
    "show_html",
    "print_banner",
    "print_rule",
    "print_ok",
    "print_warn",
    "print_err",
    "print_info",
    "print_metric_table",
    "print_gate",
    "setup_matplotlib",
    "get_project_root",
    "save_fig",
]
