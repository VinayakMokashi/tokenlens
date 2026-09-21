import json
from datetime import datetime, timezone

import pytest

from conftest import (
    CWD,
    SESSION_ID,
    assistant_line,
    text_block,
    tool_use_block,
    usage,
    write_jsonl,
)
from tokenlens.parser import (
    normalize_project_path,
    parse_entries,
    parse_session,
    parse_timestamp,
    tool_target,
)


def test_parse_timestamp_handles_zulu_and_offsets():
    ts = parse_timestamp("2026-07-27T11:54:15.562Z")
    assert ts == datetime(2026, 7, 27, 11, 54, 15, 562000, tzinfo=timezone.utc)
    assert parse_timestamp("2026-07-27T13:54:15.562+02:00") == ts
    assert parse_timestamp(None) is None
    assert parse_timestamp("not a date") is None


def test_normalize_project_path():
    assert normalize_project_path("c:\\Users\\Me\\Proj") == "C:/Users/Me/Proj"
    assert normalize_project_path("C:/Users/Me/Proj/") == "C:/Users/Me/Proj"
    assert normalize_project_path("/home/me/proj") == "/home/me/proj"
    assert normalize_project_path("") == ""


def test_groups_lines_by_message_id(sample_entries):
    session = parse_entries(sample_entries)
    # msg_a is split across two lines but is one billed call.
    assert [t.message_id for t in session.turns] == ["msg_a", "msg_b", "msg_c", "msg_d", "msg_e", "msg_f"]
    first = session.turns[0]
    assert first.index == 1
    # The later line of a message carries the final output count.
    assert first.usage.output_tokens == 160
    assert first.usage.cache_write_1h_tokens == 12_000
    assert first.usage.cache_write_5m_tokens == 0
    assert first.text_chars == len("Let me look at the file first.")
    assert [c.name for c in first.tool_calls] == ["Read"]
    assert first.stop_reason == "tool_use"
    assert first.effort == "high"
    assert first.request_id == "req_msg_a"


def test_tool_results_are_correlated(sample_entries):
    session = parse_entries(sample_entries)
    read_call = session.turns[0].tool_calls[0]
    assert read_call.target == "C:/proj/app.py"
    assert read_call.result_chars == 30_000
    assert read_call.is_error is False

    bash_call = session.turns[1].tool_calls[0]
    assert bash_call.name == "Bash"
    assert bash_call.target == "pytest -q"
    assert bash_call.is_error is True
    assert bash_call.result_preview == "1 failed, 3 passed"


def test_costs_are_priced_per_turn_model(sample_entries):
    session = parse_entries(sample_entries)
    opus_turn = session.turns[0]
    # 12,000 1h cache-write tokens on Opus 5: 12,000 / 1M * $10 = $0.12
    assert opus_turn.cost.cache_write_cost == pytest.approx(0.12)
    assert opus_turn.pricing_is_estimate is False

    sonnet_turn = session.turns[-1]
    assert sonnet_turn.model == "claude-sonnet-5"
    # 5,000 cache-read tokens on Sonnet 5: 5,000 / 1M * $0.20 = $0.001
    assert sonnet_turn.cost.cache_read_cost == pytest.approx(0.001)
    assert sonnet_turn.cost.output_cost == pytest.approx(200 / 1e6 * 10.0)

    assert session.models == ["claude-opus-5", "claude-sonnet-5"]
    assert session.total_cost == pytest.approx(sum(t.cost.total for t in session.turns))


def test_metadata_is_captured(sample_entries):
    session = parse_entries(sample_entries)
    assert session.project_path == "C:/Users/Someone/Projects/demo-app"
    assert session.title == "Fix failing test in app.py"
    assert session.first_prompt == "Please fix the failing test in app.py"
    assert session.user_prompt_count == 1  # tool_result lines are not prompts
    assert session.git_branch == "main"
    assert session.entrypoint == "cli"
    assert session.claude_versions == ["2.1.263"]
    assert session.is_subagent is False


def test_compactions_and_api_errors(sample_entries):
    session = parse_entries(sample_entries)
    assert len(session.compactions) == 1
    compaction = session.compactions[0]
    assert compaction.pre_tokens == 40_000
    assert compaction.post_tokens == 5_000
    assert compaction.dropped_tokens == 35_000
    assert compaction.trigger == "auto"
    assert compaction.after_turn == 5

    assert len(session.api_errors) == 1
    error = session.api_errors[0]
    assert error.status == 429
    assert error.error == "rate_limit"
    assert "session limit" in error.message
    # The synthetic error line never becomes a turn.
    assert all(t.model != "<synthetic>" for t in session.turns)


def test_durations(sample_entries):
    session = parse_entries(sample_entries)
    assert session.started_at == datetime(2026, 9, 1, 10, 0, 5, tzinfo=timezone.utc)
    assert session.ended_at == datetime(2026, 9, 1, 10, 3, 0, tzinfo=timezone.utc)
    assert session.wall_duration.total_seconds() == 175
    # Every gap is under 30 minutes, so active time equals wall time.
    assert session.active_duration().total_seconds() == 175


def test_active_duration_skips_long_gaps():
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"), usage(output_tokens=1)),
        assistant_line("m2", "2026-09-01T10:05:00Z", text_block("b"), usage(output_tokens=1)),
        # Resumed a week later.
        assistant_line("m3", "2026-09-08T10:00:00Z", text_block("c"), usage(output_tokens=1)),
        assistant_line("m4", "2026-09-08T10:02:00Z", text_block("d"), usage(output_tokens=1)),
    ]
    session = parse_entries(entries)
    assert session.wall_duration.days == 7
    assert session.active_duration().total_seconds() == 7 * 60


def test_usage_without_ttl_split_defaults_to_five_minute_tier():
    entries = [assistant_line("m1", "2026-09-01T10:00:00Z", text_block("a"),
                              usage(cache_5m=4_000, split=False))]
    turn = parse_entries(entries).turns[0]
    assert turn.usage.cache_write_5m_tokens == 4_000
    assert turn.usage.cache_write_1h_tokens == 0


def test_malformed_lines_are_counted_not_fatal(tmp_path):
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("ok"), usage(output_tokens=5)),
        "{this is not json",
        "",
        json.dumps({"type": "queue-operation", "operation": "enqueue"}),
    ]
    path = write_jsonl(tmp_path / "s.jsonl", entries)
    session = parse_session(str(path))
    assert len(session.turns) == 1
    assert session.skipped_lines == 1


def test_non_utf8_bytes_do_not_abort(tmp_path):
    path = tmp_path / "s.jsonl"
    line = json.dumps(assistant_line("m1", "2026-09-01T10:00:00Z", text_block("caf\u00e9"), usage(output_tokens=5)))
    # Write in cp1252 so the é is a single byte that is invalid UTF-8.
    path.write_bytes(line.encode("cp1252") + b"\n")
    session = parse_session(str(path))
    assert len(session.turns) == 1


def test_subagents_are_attached_to_parent(tmp_path):
    project_dir = tmp_path / "C--Users-Someone-Projects-demo-app"
    main_path = write_jsonl(project_dir / f"{SESSION_ID}.jsonl", [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("main"), usage(output_tokens=1_000_000)),
    ])
    sub_entries = [assistant_line("s1", "2026-09-01T10:00:30Z", text_block("sub"),
                                  usage(output_tokens=1_000_000), model="claude-haiku-4-5")]
    for e in sub_entries:
        e["isSidechain"] = True
        e["agentId"] = "abc123"
    write_jsonl(project_dir / SESSION_ID / "subagents" / "agent-abc123.jsonl", sub_entries)

    session = parse_session(str(main_path))
    assert len(session.subagents) == 1
    sub = session.subagents[0]
    assert sub.is_subagent is True
    assert sub.agent_id == "abc123"
    assert sub.parent_session_id == SESSION_ID
    assert sub.project_path == session.project_path == normalize_project_path(CWD)
    # $25 of Opus output in the main session plus $5 of Haiku output in the subagent.
    assert session.total_cost == pytest.approx(25.0)
    assert session.subagent_cost == pytest.approx(5.0)
    assert session.total_cost_with_subagents == pytest.approx(30.0)
    assert len(session.all_turns()) == 2


def test_workflow_subagents_are_found_two_levels_deep(tmp_path):
    project_dir = tmp_path / "proj"
    main_path = write_jsonl(project_dir / f"{SESSION_ID}.jsonl", [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("main"), usage(output_tokens=10)),
    ])
    wf_dir = project_dir / SESSION_ID / "subagents" / "workflows" / "wf_deadbeef-123"
    write_jsonl(wf_dir / "agent-a1.jsonl", [
        assistant_line("s1", "2026-09-01T10:00:30Z", text_block("sub"), usage(output_tokens=10)),
    ])
    write_jsonl(wf_dir / "agent-a2.jsonl", [
        assistant_line("s2", "2026-09-01T10:00:31Z", text_block("sub"), usage(output_tokens=10)),
    ])
    write_jsonl(project_dir / SESSION_ID / "subagents" / "agent-direct.jsonl", [
        assistant_line("s3", "2026-09-01T10:00:32Z", text_block("sub"), usage(output_tokens=10)),
    ])
    session = parse_session(str(main_path))
    assert len(session.subagents) == 3
    by_agent = {s.agent_id: s for s in session.subagents}
    assert by_agent["a1"].workflow_id == "wf_deadbeef-123"
    assert by_agent["a2"].workflow_id == "wf_deadbeef-123"
    assert by_agent["direct"].workflow_id == ""


def test_subagents_not_parsed_when_disabled(tmp_path):
    project_dir = tmp_path / "proj"
    main_path = write_jsonl(project_dir / f"{SESSION_ID}.jsonl", [
        assistant_line("m1", "2026-09-01T10:00:00Z", text_block("main"), usage(output_tokens=10)),
    ])
    write_jsonl(project_dir / SESSION_ID / "subagents" / "agent-x.jsonl", [
        assistant_line("s1", "2026-09-01T10:00:30Z", text_block("sub"), usage(output_tokens=10)),
    ])
    assert parse_session(str(main_path), include_subagents=False).subagents == []


@pytest.mark.parametrize(
    "name, tool_input, expected",
    [
        ("Read", {"file_path": "/a/b.py"}, "/a/b.py"),
        ("Read", {"file_path": "/a/b.py", "offset": 10, "limit": 50}, "/a/b.py [offset=10 limit=50]"),
        ("Bash", {"command": "ls -la\ncat x", "description": "list"}, "ls -la"),
        ("Grep", {"pattern": "TODO", "path": "src"}, "TODO in src"),
        ("Glob", {"pattern": "**/*.py"}, "**/*.py"),
        ("WebFetch", {"url": "https://example.com"}, "https://example.com"),
        ("Agent", {"subagent_type": "Explore", "description": "find callers"}, "Explore: find callers"),
        ("TodoWrite", {"todos": [1, 2, 3]}, "3 items"),
        ("Mystery", {"description": "do a thing"}, "do a thing"),
        ("Mystery", {"foo": 1}, '{"foo": 1}'),
        ("Mystery", "not a dict", ""),
    ],
)
def test_tool_target(name, tool_input, expected):
    assert tool_target(name, tool_input) == expected


def test_turn_kind_classification():
    entries = [
        assistant_line("m1", "2026-09-01T10:00:00Z", tool_use_block("t", "Bash", {"command": "ls"}), usage()),
        assistant_line("m2", "2026-09-01T10:00:01Z", {"type": "thinking", "thinking": "hmm"}, usage()),
        assistant_line("m3", "2026-09-01T10:00:02Z", text_block("hello"), usage()),
    ]
    kinds = [t.kind for t in parse_entries(entries).turns]
    assert kinds == ["tool_use", "thinking", "text"]
