import pytest

from conftest import (
    api_error_line,
    assistant_line,
    text_block,
    thinking_block,
    tool_result_line,
    tool_use_block,
    usage,
)
from tokenlens.findings import Thresholds, detect_findings, total_estimated_savings
from tokenlens.parser import parse_entries


def _rules(findings):
    return [f.rule for f in findings]


def test_sample_session_findings(sample_entries):
    session = parse_entries(sample_entries)
    findings = detect_findings(session)
    rules = _rules(findings)

    # Turn 3 re-reads app.py without an edit in between; turn 5 re-reads
    # after the Edit in turn 4 and must NOT be flagged.
    dups = [f for f in findings if f.rule == "duplicate_read"]
    assert len(dups) == 1
    assert dups[0].turns == [1, 3]
    assert "turn 1" in dups[0].detail

    # Each 30,000-char Read is a large result.
    large = [f for f in findings if f.rule == "large_tool_result"]
    assert [f.turns[0] for f in large] == [1, 3, 5]
    # Turn 1 is carried through turns 2-5 (4 turns); turn 5 is carried by nobody.
    assert large[0].evidence["carry_turns"] == 4
    assert large[-1].evidence["carry_turns"] == 0
    assert large[0].est_dollars_saved > large[-1].est_dollars_saved

    assert "api_errors" in rules
    api = next(f for f in findings if f.rule == "api_errors")
    assert "1x 429 rate_limit" in api.detail

    # No bloat, no error streak, no estimated pricing in the sample.
    assert "context_bloat" not in rules
    assert "error_streak" not in rules
    assert "estimated_pricing" not in rules

    # Every finding carries provenance.
    assert all(f.session_id == session.session_id for f in findings)
    assert all(f.project_path == session.project_path for f in findings)

    # Sorted by severity, then dollars.
    ranks = [f.severity_rank for f in findings]
    assert ranks == sorted(ranks, reverse=True)


def test_context_bloat_estimates_excess_carry():
    entries = []
    for i in range(1, 6):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:00:{i:02d}Z", text_block("x"),
            usage(input_tokens=10, output_tokens=50, cache_read=500_000, cache_1h=1_000)))
    session = parse_entries(entries)
    th = Thresholds(context_bloat_tokens=400_000, healthy_context_tokens=60_000)
    findings = detect_findings(session, thresholds=th)
    bloat = [f for f in findings if f.rule == "context_bloat"]
    assert len(bloat) == 1
    b = bloat[0]
    assert b.turns == [1, 5]
    # Excess = (500,000 - 60,000) * 5 turns at Opus 5 cache-read $0.50/M
    assert b.est_dollars_saved == pytest.approx(440_000 * 5 / 1e6 * 0.5)
    assert b.est_tokens_saved == 440_000 * 5
    assert b.severity == "warning"
    assert "/compact" in b.detail


def test_context_bloat_becomes_critical_above_ten_dollars():
    entries = []
    for i in range(1, 60):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:{i:02d}:00Z", text_block("x"),
            usage(input_tokens=10, output_tokens=50, cache_read=900_000, cache_1h=1_000)))
    findings = detect_findings(parse_entries(entries))
    bloat = next(f for f in findings if f.rule == "context_bloat")
    assert bloat.severity == "critical"
    assert findings[0] is bloat  # critical sorts first


def test_cache_expiry_detected_after_gap():
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                       usage(input_tokens=10, output_tokens=10, cache_1h=50_000)),
        assistant_line("m2", "2026-09-01T10:01:00Z", text_block("b"),
                       usage(input_tokens=10, output_tokens=10, cache_read=50_000, cache_1h=500)),
        # Two hours later: nothing read from cache, everything re-written.
        assistant_line("m3", "2026-09-01T12:30:00Z", text_block("c"),
                       usage(input_tokens=10, output_tokens=10, cache_read=0, cache_1h=52_000)),
    ]
    findings = detect_findings(parse_entries(entries))
    expiry = [f for f in findings if f.rule == "cache_expired"]
    assert len(expiry) == 1
    assert expiry[0].turns == [3]
    assert "2.5 hour gap" in expiry[0].detail
    # Paid 52,000 * $10/M = $0.52 instead of 52,000 * $0.50/M = $0.026
    assert expiry[0].est_dollars_saved == pytest.approx(0.52 - 0.026)


def test_first_turn_is_never_a_cache_miss():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                              usage(input_tokens=10, output_tokens=10, cache_1h=500_000))]
    assert "cache_expired" not in _rules(detect_findings(parse_entries(entries)))


def test_error_streak():
    entries = []
    for i in range(1, 5):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:00:{i:02d}Z",
            tool_use_block(f"t{i}", "Bash", {"command": "make build"}), usage(output_tokens=20)))
        entries.append(tool_result_line(f"2026-09-01T10:00:{i:02d}.500Z", f"t{i}", "error: no rule", is_error=True))
    findings = detect_findings(parse_entries(entries))
    streak = [f for f in findings if f.rule == "error_streak"]
    assert len(streak) == 1
    assert streak[0].turns == [1, 2, 3, 4]
    assert streak[0].evidence["count"] == 4
    assert "make build" in streak[0].detail


def test_error_streak_needs_consecutive_failures():
    entries = []
    for i in range(1, 5):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:00:{i:02d}Z",
            tool_use_block(f"t{i}", "Bash", {"command": "x"}), usage(output_tokens=20)))
        # Alternate failure and success.
        entries.append(tool_result_line(f"2026-09-01T10:00:{i:02d}.500Z", f"t{i}", "out", is_error=(i % 2 == 1)))
    assert "error_streak" not in _rules(detect_findings(parse_entries(entries)))


def test_thinking_share():
    entries = [assistant_line(
        "m1", "2026-09-01T10:00:00Z", thinking_block("t" * 40_000),
        usage(output_tokens=12_000), effort="xhigh")]
    findings = detect_findings(parse_entries(entries))
    think = [f for f in findings if f.rule == "thinking_share"]
    assert len(think) == 1
    assert "83%" in think[0].title  # 10,000 est. thinking tokens / 12,000 output
    assert "xhigh: 1 turns" in think[0].detail


def test_thinking_share_ignores_tiny_sessions():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", thinking_block("t" * 4_000), usage(output_tokens=1_000))]
    assert "thinking_share" not in _rules(detect_findings(parse_entries(entries)))


def test_estimated_pricing_flagged():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                              usage(output_tokens=10), model="claude-opus-12")]
    findings = detect_findings(parse_entries(entries))
    est = [f for f in findings if f.rule == "estimated_pricing"]
    assert len(est) == 1
    assert "claude-opus-12" in est[0].detail
    assert "--pricing" in est[0].detail


def test_api_errors_summary():
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"), usage(output_tokens=10)),
        api_error_line("2026-09-01T10:00:05Z", status=429),
        api_error_line("2026-09-01T10:00:06Z", status=529, error="overloaded"),
    ]
    findings = detect_findings(parse_entries(entries))
    api = next(f for f in findings if f.rule == "api_errors")
    assert "2 API errors" in api.title
    assert "1x 429 rate_limit" in api.detail and "1x 529 overloaded" in api.detail


def test_total_estimated_savings_avoids_double_counting(sample_entries):
    session = parse_entries(sample_entries)
    findings = detect_findings(session)
    # No bloat finding: every dollar estimate counts.
    assert total_estimated_savings(findings) == pytest.approx(sum(f.est_dollars_saved for f in findings))

    # With a bloat finding, per-result rules are excluded.
    entries = []
    for i in range(1, 4):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:00:{i:02d}Z",
            tool_use_block(f"t{i}", "Read", {"file_path": f"/f{i}.py"}),
            usage(input_tokens=10, output_tokens=50, cache_read=500_000, cache_1h=10_000)))
        entries.append(tool_result_line(f"2026-09-01T10:00:{i:02d}.500Z", f"t{i}", "y" * 40_000))
    findings = detect_findings(parse_entries(entries))
    rules = _rules(findings)
    assert "context_bloat" in rules and "large_tool_result" in rules
    bloat = next(f for f in findings if f.rule == "context_bloat")
    assert total_estimated_savings(findings) == pytest.approx(bloat.est_dollars_saved)
