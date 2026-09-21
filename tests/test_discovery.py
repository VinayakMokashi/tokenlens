import os
from pathlib import Path

import pytest

from conftest import SESSION_ID, assistant_line, text_block, usage, write_jsonl
from tokenlens.discovery import (
    find_session,
    find_sessions,
    latest_session,
    load_sessions,
    projects_root,
    resolve_target,
)


def _session(path: Path, message_id: str, ts: str, **kwargs) -> Path:
    return write_jsonl(path, [assistant_line(message_id, ts, text_block("x"), usage(output_tokens=100), **kwargs)])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "projects"
    proj_a = base / "C--Users-Someone-Projects-demo-app"
    proj_b = base / "-home-me-other"
    _session(proj_a / f"{SESSION_ID}.jsonl", "m1", "2026-09-01T10:00:00Z")
    _session(proj_a / SESSION_ID / "subagents" / "agent-aaa.jsonl", "s1", "2026-09-01T10:00:30Z")
    _session(proj_a / SESSION_ID / "subagents" / "agent-bbb.jsonl", "s2", "2026-09-01T10:00:40Z")
    _session(proj_a / SESSION_ID / "subagents" / "workflows" / "wf_1234abcd-56e" / "agent-www.jsonl",
             "s4", "2026-09-01T10:00:50Z")
    _session(proj_b / "99999999-0000-0000-0000-000000000000.jsonl", "m2", "2026-09-05T10:00:00Z")
    # Orphaned subagent whose parent transcript is gone.
    _session(proj_b / "deadbeef-0000-0000-0000-000000000000" / "subagents" / "agent-ccc.jsonl",
             "s3", "2026-09-03T10:00:00Z")
    # Make mtimes deterministic: proj_b's main session is newest.
    os.utime(proj_a / f"{SESSION_ID}.jsonl", (1_000, 1_000))
    for sub in (proj_a / SESSION_ID / "subagents").rglob("*.jsonl"):
        os.utime(sub, (1_500, 1_500))
    os.utime(proj_b / "99999999-0000-0000-0000-000000000000.jsonl", (3_000, 3_000))
    os.utime(proj_b / "deadbeef-0000-0000-0000-000000000000" / "subagents" / "agent-ccc.jsonl", (500, 500))
    return base


def test_find_sessions_attaches_subagents_and_orders_newest_first(root):
    refs = find_sessions(root)
    assert [r.session_id[:8] for r in refs] == ["99999999", "11111111", "ccc"]
    main = refs[1]
    assert len(main.subagent_paths) == 3
    assert [p.name for p in main.subagent_paths] == ["agent-aaa.jsonl", "agent-bbb.jsonl", "agent-www.jsonl"]
    assert main.is_subagent is False
    # The subagent files are newer than the main file and count toward recency.
    assert main.latest_mtime == 1_500
    assert main.total_size > main.size

    orphan = refs[2]
    assert orphan.is_subagent is True
    assert orphan.parent_session_id == "deadbeef-0000-0000-0000-000000000000"


def test_find_sessions_missing_root_returns_empty(tmp_path):
    assert find_sessions(tmp_path / "nope") == []


def test_find_session_by_prefix(root):
    assert find_session("1111", root).session_id == SESSION_ID
    assert find_session("9999", root).session_id.startswith("99999999")
    assert find_session("zzzz", root) is None
    assert find_session("", root) is None


def test_latest_session_skips_orphans(root):
    latest = latest_session(root)
    assert latest.session_id.startswith("99999999")
    filtered = latest_session(root, project_filter=lambda r: r.project_dir.startswith("C--"))
    assert filtered.session_id == SESSION_ID


def test_resolve_target(root):
    file_path = root / "-home-me-other" / "99999999-0000-0000-0000-000000000000.jsonl"
    assert resolve_target(str(file_path), root) == file_path
    assert resolve_target("latest", root) == file_path
    assert resolve_target("1111", root).name == f"{SESSION_ID}.jsonl"
    with pytest.raises(FileNotFoundError):
        resolve_target("does-not-exist", root)


def test_load_sessions_includes_subagent_cost(root):
    refs = find_sessions(root)
    sessions = {s.session_id[:8]: s for s in load_sessions(refs)}
    main = sessions["11111111"]
    assert len(main.subagents) == 3
    assert main.subagent_cost == pytest.approx(3 * 100 / 1e6 * 25.0)
    workflow_agents = [s for s in main.subagents if s.workflow_id]
    assert [s.workflow_id for s in workflow_agents] == ["wf_1234abcd-56e"]
    assert workflow_agents[0].agent_id == "www"
    assert main.project_dir == "C--Users-Someone-Projects-demo-app"
    from tokenlens.analysis import aggregate

    report = aggregate(list(sessions.values()))
    # Every API call behind the spend total is counted: 3 main-session turns
    # plus 3 subagent turns (the orphan subagent counts as its own session).
    assert report.turn_count == 3
    assert report.subagent_turn_count == 3
    assert report.total_turn_count == 6
    assert report.total_turn_count == sum(m.turns for m in report.by_model)

    orphan = sessions["ccc"]
    assert orphan.is_subagent is True
    assert orphan.parent_session_id == "deadbeef-0000-0000-0000-000000000000"


def test_projects_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENLENS_PROJECTS_DIR", str(tmp_path))
    assert projects_root() == tmp_path
    assert projects_root("/explicit") == Path("/explicit")
    monkeypatch.delenv("TOKENLENS_PROJECTS_DIR")
    assert projects_root().name == "projects"
