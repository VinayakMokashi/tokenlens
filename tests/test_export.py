import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from tokenlens.analysis import analyze_session
from tokenlens.export import jsonable, render_markdown, session_report_to_dict, to_json
from tokenlens.findings import detect_findings
from tokenlens.formatting import duration, money, pct, shorten, table, tokens
from tokenlens.parser import parse_entries


def test_jsonable_handles_common_types():
    stamp = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    assert jsonable(stamp) == "2026-09-01T10:00:00+00:00"
    assert jsonable(date(2026, 9, 1)) == "2026-09-01"
    assert jsonable(timedelta(minutes=2)) == 120.0
    assert jsonable(Path("a/b")) == "a/b"
    assert jsonable({"k": [1, (2, 3)]}) == {"k": [1, [2, 3]]}


def test_session_dict_round_trips_through_json(sample_entries):
    session = parse_entries(sample_entries)
    report = analyze_session(session)
    data = session_report_to_dict(report, detect_findings(session))
    text = to_json(data)
    again = json.loads(text)
    assert again["cost"]["total"] == data["cost"]["total"]
    assert again["usage"]["context_tokens"] == session.usage.context_tokens
    assert again["turns"][1]["tool_calls"][0]["is_error"] is True
    assert again["by_kind"]["tool_use"]["turns"] == 5


def test_markdown_contains_sections(sample_entries):
    session = parse_entries(sample_entries)
    md = render_markdown(analyze_session(session), detect_findings(session))
    for needle in ("# Fix failing test", "## Cost", "## Same conversation on another model", "## Findings",
                   "**WARNING: Re-read an unchanged file"):
        assert needle in md


def test_money_formatting_is_adaptive():
    assert money(0) == "$0.00"
    assert money(0.0042) == "$0.0042"
    assert money(0.042) == "$0.042"
    assert money(4.2) == "$4.20"
    assert money(420.0) == "$420.00"
    assert money(4200.0) == "$4,200"
    assert money(None) == "-"
    assert money(1.23456, precision=4) == "$1.2346"


def test_tokens_and_pct_and_duration():
    assert tokens(999) == "999"
    assert tokens(1_500) == "1.5K"
    assert tokens(25_000) == "25K"
    assert tokens(2_500_000) == "2.50M"
    assert pct(0.1234) == "12%"
    assert pct(0.1234, 1) == "12.3%"
    assert duration(timedelta(seconds=45)) == "45s"
    assert duration(timedelta(minutes=5, seconds=7)) == "5m 07s"
    assert duration(timedelta(hours=3, minutes=2)) == "3h 02m"
    assert duration(timedelta(days=2, hours=1)) == "2d 1h"
    assert duration(None) == "-"


def test_table_and_shorten():
    out = table(["Name", "Count"], [["a", 1], ["bbb", 22]])
    lines = out.splitlines()
    assert lines[0].startswith("Name")
    assert lines[2].endswith(" 1")   # numeric column right-aligned
    assert lines[3].endswith("22")
    assert table(["x"], []) == "(none)"
    assert shorten("a  b   c", 10) == "a b c"
    assert shorten("abcdefghijk", 8) == "abcde..."
