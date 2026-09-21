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


def test_different_slices_of_one_file_are_not_duplicates():
    entries = []
    for i, (offset, limit) in enumerate(((0, 200), (200, 200), (0, 200)), start=1):
        entries.append(assistant_line(
            f"m{i}", f"2026-09-01T10:00:{i:02d}Z",
            tool_use_block(f"t{i}", "Read", {"file_path": "/big.py", "offset": offset, "limit": limit}),
            usage(output_tokens=20)))
        entries.append(tool_result_line(f"2026-09-01T10:00:{i:02d}.500Z", f"t{i}", "z" * 5_000))
    findings = detect_findings(parse_entries(entries))
    dups = [f for f in findings if f.rule == "duplicate_read"]
    # Turn 2 reads a different slice: fine. Turn 3 repeats turn 1's slice: flagged.
    assert len(dups) == 1
    assert dups[0].turns == [1, 3]


def test_thinking_share_never_exceeds_100_percent():
    # Dense text: 40K chars of thinking against only 8K output tokens.
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", thinking_block("t" * 40_000),
                              usage(output_tokens=8_000), effort="max")]
    think = next(f for f in detect_findings(parse_entries(entries)) if f.rule == "thinking_share")
    assert "About 100%" in think.title
    assert think.evidence["thinking_tokens"] == 8_000


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
    expiry = [f for f in findings if f.rule == "cache_miss"]
    assert len(expiry) == 1
    assert expiry[0].turns == [3]
    assert "2.5 hours after the previous call" in expiry[0].detail
    assert expiry[0].evidence["gap_minutes"] == 149.0
    # Paid 52,000 * $10/M = $0.52 instead of 52,000 * $0.50/M = $0.026. The
    # extra is explained but not claimed as avoidable: after a long break
    # the re-write cannot be prevented.
    assert expiry[0].evidence["extra_cost"] == pytest.approx(0.52 - 0.026)
    assert expiry[0].est_dollars_saved == 0.0
    assert total_estimated_savings(findings) == 0.0


def test_model_switch_is_reported_separately_from_cache_expiry():
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                       usage(input_tokens=10, output_tokens=10, cache_1h=50_000)),
        # Two minutes later on a different model: caches are per model.
        assistant_line("m2", "2026-09-01T10:02:00Z", text_block("b"),
                       usage(input_tokens=10, output_tokens=10, cache_read=0, cache_1h=50_500),
                       model="claude-sonnet-5"),
    ]
    findings = detect_findings(parse_entries(entries))
    rules = _rules(findings)
    assert "cache_miss" not in rules
    switch = next(f for f in findings if f.rule == "model_switch")
    assert switch.turns == [2]
    assert "claude-opus-5 to claude-sonnet-5" in switch.detail
    assert switch.est_dollars_saved == 0.0


def test_first_turn_is_never_a_cache_miss():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                              usage(input_tokens=10, output_tokens=10, cache_1h=500_000))]
    assert "cache_miss" not in _rules(detect_findings(parse_entries(entries)))


def test_cache_miss_detected_even_when_shared_prefix_is_read():
    # Claude Code keeps a ~28K system/tools prefix cached across sessions, so
    # a miss on the conversation still reads that prefix. Real case: turn 183
    # of a session read 28,023 and re-wrote 371,704 ten minutes after turn 182.
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                       usage(input_tokens=5, output_tokens=10, cache_read=28_000, cache_1h=372_000)),
        assistant_line("m2", "2026-09-01T10:00:10Z", text_block("b"),
                       usage(input_tokens=5, output_tokens=10, cache_read=400_000, cache_1h=500)),
        assistant_line("m3", "2026-09-01T10:10:10Z", text_block("c"),
                       usage(input_tokens=5, output_tokens=10, cache_read=28_000, cache_1h=372_600)),
    ]
    findings = detect_findings(parse_entries(entries))
    misses = [f for f in findings if f.rule == "cache_miss"]
    assert [f.turns for f in misses] == [[3]]
    miss = misses[0]
    assert "only 10 minutes after the previous call" in miss.detail
    assert "invalidated rather than expired" in miss.detail
    # 372,600 1h-write tokens on Opus 5 cost $3.726; as reads they'd cost $0.1863.
    assert miss.evidence["extra_cost"] == pytest.approx(3.726 - 0.1863)
    assert "~$3.54 extra" in miss.title
    assert miss.est_dollars_saved == 0.0


def test_large_new_content_is_not_a_cache_miss():
    # Turn 2 reads all of turn 1's context and adds a big tool result: a large
    # write, but the cache was reused, so it is not a miss.
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                       usage(input_tokens=5, output_tokens=10, cache_1h=12_000)),
        assistant_line("m2", "2026-09-01T10:00:10Z", text_block("b"),
                       usage(input_tokens=5, output_tokens=10, cache_read=12_000, cache_1h=40_000)),
    ]
    assert "cache_miss" not in _rules(detect_findings(parse_entries(entries)))


def test_bloat_run_is_not_split_by_a_small_dip():
    contexts = [450_000, 460_000, 399_700, 470_000, 480_000, 300_000]
    entries = [assistant_line(f"m{i}", f"2026-09-01T10:00:{i:02d}Z", text_block("x"),
                              usage(input_tokens=5, output_tokens=10, cache_read=c - 1_000, cache_1h=995))
               for i, c in enumerate(contexts, start=1)]
    bloat = [f for f in detect_findings(parse_entries(entries)) if f.rule == "context_bloat"]
    assert [f.turns for f in bloat] == [[1, 5]]


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
    bloated = parse_entries(entries)
    findings = detect_findings(bloated)
    rules = _rules(findings)
    assert "context_bloat" in rules and "large_tool_result" in rules
    bloat = next(f for f in findings if f.rule == "context_bloat")
    assert total_estimated_savings(findings) == pytest.approx(bloat.est_dollars_saved)

    # Pooled across sessions, a bloat finding in one session must not hide
    # the per-result savings of another session.
    other = parse_entries(sample_entries, path="/tmp/other.jsonl")
    other_findings = detect_findings(other)
    pooled = findings + other_findings
    assert total_estimated_savings(pooled) == pytest.approx(
        bloat.est_dollars_saved + total_estimated_savings(other_findings))
