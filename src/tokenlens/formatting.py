"""Small, dependency-free formatting helpers shared by the text report,
the Markdown export, and the web templates."""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Sequence


def money(value: Optional[float], precision: Optional[int] = None) -> str:
    """Adaptive dollar formatting: cents for small values, whole dollars
    for large ones, so columns stay readable across five orders of
    magnitude."""
    if value is None:
        return "-"
    if precision is not None:
        return f"${value:,.{precision}f}"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"${value:,.0f}"
    if magnitude >= 1:
        return f"${value:,.2f}"
    if magnitude >= 0.01:
        return f"${value:.3f}"
    if magnitude == 0:
        return "$0.00"
    return f"${value:.4f}"


def signed_money(value: float) -> str:
    text = money(abs(value))
    return f"-{text}" if value < 0 else f"+{text}"


def tokens(value: Optional[int]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}K"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def pct(fraction: Optional[float], digits: int = 0) -> str:
    if fraction is None:
        return "-"
    return f"{fraction * 100:.{digits}f}%"


def duration(delta: Optional[timedelta]) -> str:
    if delta is None:
        return "-"
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def when(stamp: Optional[datetime], with_time: bool = True) -> str:
    if stamp is None:
        return "-"
    local = stamp.astimezone()
    return local.strftime("%Y-%m-%d %H:%M") if with_time else local.strftime("%Y-%m-%d")


def shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: max(width - 3, 1)] + "..."


def wrap(text: str, width: int = 78, indent: str = "") -> str:
    return textwrap.fill(" ".join(text.split()), width=width, initial_indent=indent, subsequent_indent=indent)


def table(headers: Sequence[str], rows: Iterable[Sequence[object]],
          align: Optional[Sequence[str]] = None, indent: str = "") -> str:
    """Render an ASCII table. ``align`` is a sequence of 'l'/'r' per column;
    numeric-looking columns default to right alignment."""
    str_rows: List[List[str]] = [[str(c) for c in row] for row in rows]
    if not str_rows:
        return indent + "(none)"
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    if align is None:
        align = []
        for i in range(len(headers)):
            sample = next((r[i] for r in str_rows if r[i] not in ("", "-")), "")
            numeric = sample.replace(",", "").replace(".", "").replace("$", "").replace("%", "")
            numeric = numeric.replace("K", "").replace("M", "").replace("+", "").replace("-", "").strip()
            align.append("r" if numeric.isdigit() else "l")

    def fmt(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if align[i] == "r" else cell.ljust(widths[i]))
        return indent + "  ".join(parts).rstrip()

    lines = [fmt(headers), indent + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def heading(text: str, width: int = 78, char: str = "=") -> str:
    return f"{text}\n{char * min(len(text), width)}"
