from datetime import date

import pytest

from conftest import assistant_line, text_block, usage
from tokenlens.analysis import (
    aggregate,
    analyze_session,
    context_series,
    phases,
    rolling_window,
    tool_stats,
    usage_by_kind,
    usage_by_model,
    what_if_costs,
)
from tokenlens.parser import parse_entries
from tokenlens.pricing import DEFAULT_TABLE


def test_usage_by_model_sorts_by_cost(sample_entries):
    session = parse_entries(sample_entries)
    by_model = usage_by_model(session.turns)
    assert [m.model for m in by_model] == ["claude-opus-5", "claude-sonnet-5"]
    assert by_model[0].turns == 5
    assert by_model[1].turns == 1
    assert by_model[0].display == "Claude Opus 5"
    assert by_model[0].cost.total == pytest.approx(sum(t.cost.total for t in session.turns[:5]))


def test_usage_by_kind(sample_entries):
    session = parse_entries(sample_entries)
    kinds = usage_by_kind(session.turns)
    assert kinds["tool_use"][0] == 5
    assert kinds["text"][0] == 1


def test_tool_stats(sample_entries):
    session = parse_entries(sample_entries)
    stats = {s.name: s for s in tool_stats(session.turns)}
    assert stats["Read"].calls == 3
    assert stats["Read"].result_chars == 90_000
    assert stats["Read"].largest_result_chars == 30_000
    assert stats["Read"].largest_target == "C:/proj/app.py"
    assert stats["Read"].est_result_tokens == 22_500
    assert stats["Bash"].errors == 1
    # Sorted by characters pulled into context.
    assert tool_stats(session.turns)[0].name == "Read"


def test_context_series_marks_compaction(sample_entries):
    session = parse_entries(sample_entries)
    series = context_series(session)
    assert [p.turn_index for p in series] == [1, 2, 3, 4, 5, 6]
    # Turn 1: 10 input + 12,000 cache write
    assert series[0].context_tokens == 12_010
    # Turn 6 follows the compaction after turn 5.
    assert series[5].compaction_before is True
    assert all(p.compaction_before is False for p in series[:5])
    assert series[1].carry_cost == pytest.approx(12_000 / 1e6 * 0.5)


def test_phases_split_at_compaction(sample_entries):
    session = parse_entries(sample_entries)
    result = phases(session)
    assert len(result) == 2
    assert (result[0].first_turn, result[0].last_turn, result[0].turns) == (1, 5, 5)
    assert (result[1].first_turn, result[1].last_turn, result[1].turns) == (6, 6, 1)
    assert result[0].peak_context == 16_705
    assert result[1].start_context == 5_020


def test_phases_without_compaction_is_single_phase():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"), usage(output_tokens=1))]
    result = phases(parse_entries(entries))
    assert len(result) == 1 and result[0].turns == 1


def test_what_if_reprices_same_tokens(sample_entries):
    session = parse_entries(sample_entries)
    what_if = {w.model: w for w in what_if_costs(session.turns)}
    actual = session.total_cost
    # Everything on Haiku is cheaper than the mixed Opus/Sonnet actual.
    assert what_if["claude-haiku-4-5"].cost < actual
    assert what_if["claude-haiku-4-5"].delta == pytest.approx(what_if["claude-haiku-4-5"].cost - actual)
    # Everything on Fable 5.1 is more expensive.
    assert what_if["claude-fable-5-1"].cost > actual
    assert what_if["claude-fable-5-1"].delta_pct > 0
    # Re-pricing at the model actually used for most turns is close to actual.
    opus = DEFAULT_TABLE.spec_for_key("claude-opus-5")
    expected = sum(DEFAULT_TABLE.price_with_rates(t.usage, opus.rates).total for t in session.turns)
    assert what_if["claude-opus-5"].cost == pytest.approx(expected)


def test_session_report_metrics(sample_entries):
    session = parse_entries(sample_entries)
    report = analyze_session(session)
    assert report.turn_count == 6
    assert report.peak_context == 16_705
    assert report.carry_cost == pytest.approx(session.cost.cache_read_cost)
    assert report.new_work_cost == pytest.approx(
        session.cost.input_cost + session.cost.cache_write_cost + session.cost.output_cost)
    assert 0 < report.carry_share < 1
    prompt_tokens = session.usage.context_tokens
    assert report.cache_hit_rate == pytest.approx(session.usage.cache_read_tokens / prompt_tokens)
    assert report.cost_per_prompt == pytest.approx(session.total_cost / 1)
    assert report.has_estimated_pricing is False
    assert [d.day for d in report.daily] == [date(2026, 9, 1)]
    assert report.daily[0].turns == 6


def test_aggregate_groups_by_project_and_model(sample_entries):
    a = parse_entries(sample_entries)
    b_entries = [assistant_line("z1", "2026-09-02T10:00:00Z", text_block("hi"),
                                usage(output_tokens=1_000_000), model="claude-haiku-4-5")]
    for e in b_entries:
        e["cwd"] = "/home/me/other"
    b = parse_entries(b_entries, path="/tmp/other.jsonl")

    report = aggregate([a, b])
    assert report.session_count == 2
    assert report.turn_count == 7
    assert report.total_cost == pytest.approx(a.total_cost + b.total_cost)
    projects = {p.project_path: p for p in report.by_project}
    assert projects["/home/me/other"].cost == pytest.approx(5.0)
    assert projects["C:/Users/Someone/Projects/demo-app"].sessions == 1
    assert {m.model for m in report.by_model} == {"claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"}
    assert [d.day for d in report.daily] == [date(2026, 9, 1), date(2026, 9, 2)]
    # Newest session first.
    assert report.sessions[0].project_path == "/home/me/other"
    assert report.sessions[1].title == "Fix failing test in app.py"


def test_filter_sessions_by_project_and_window(sample_entries):
    from datetime import datetime, timezone

    from tokenlens.analysis import filter_sessions

    a = parse_entries(sample_entries)
    b_entries = [assistant_line("z1", "2026-08-01T10:00:00Z", text_block("hi"), usage(output_tokens=10))]
    for e in b_entries:
        e["cwd"] = "/home/me/other"
    b = parse_entries(b_entries, path="/tmp/other.jsonl")
    now = datetime(2026, 9, 3, tzinfo=timezone.utc)

    assert filter_sessions([a, b]) == [a, b]
    assert filter_sessions([a, b], project="DEMO-APP") == [a]
    assert filter_sessions([a, b], days=7, now=now) == [a]
    assert filter_sessions([a, b], days=60, now=now) == [a, b]
    assert filter_sessions([a, b], project="other", days=7, now=now) == []


def test_rolling_window():
    entries = []
    for day in range(1, 11):
        entries.append(assistant_line(f"m{day}", f"2026-09-{day:02d}T10:00:00Z", text_block("a"), usage(output_tokens=1)))
    report = analyze_session(parse_entries(entries))
    last3 = rolling_window(report.daily, 3)
    assert [d.day.day for d in last3] == [8, 9, 10]
    assert rolling_window([], 7) == []
