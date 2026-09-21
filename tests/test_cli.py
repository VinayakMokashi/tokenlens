import csv
import json

import pytest

from conftest import SESSION_ID
from tokenlens.cli import main


@pytest.fixture
def projects_dir(sample_session_path):
    # sample_session_path is <tmp>/projects/<encoded>/<session>.jsonl
    return sample_session_path.parent.parent


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_analyze_by_path_renders_text_report(capsys, sample_session_path):
    code, out, err = run(capsys, "analyze", sample_session_path)
    assert code == 0
    assert "Session: Fix failing test in app.py" in out
    assert "Project:   C:/Users/Someone/Projects/demo-app" in out
    assert "Cache read (carrying context)" in out
    assert "Same conversation on another model" in out
    assert "Findings" in out
    assert "Re-read an unchanged file" in out
    assert "API-equivalent" in out


def test_analyze_latest_and_prefix(capsys, projects_dir):
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "analyze", "latest")
    assert code == 0 and SESSION_ID in out
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "analyze", SESSION_ID[:6])
    assert code == 0 and SESSION_ID in out


def test_analyze_json_is_valid_and_complete(capsys, sample_session_path):
    code, out, _ = run(capsys, "analyze", sample_session_path, "--json")
    assert code == 0
    data = json.loads(out)
    assert data["schema_version"] == 1
    assert data["session"]["session_id"] == SESSION_ID
    assert len(data["turns"]) == 6
    assert data["turns"][0]["usage"]["cache_write_1h_tokens"] == 12_000
    assert data["turns"][0]["cost"]["cache_write_1h_cost"] == pytest.approx(0.12)
    assert data["cost"]["cache_write_cost"] == pytest.approx(data["cost"]["cache_write_1h_cost"])
    assert data["metrics"]["peak_context"] == 16_705
    assert {f["rule"] for f in data["findings"]} >= {"duplicate_read", "large_tool_result", "api_errors"}
    assert [w["model"] for w in data["what_if"]] == [
        "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
    assert data["compactions"][0]["pre_tokens"] == 40_000


def test_analyze_json_without_turns(capsys, sample_session_path):
    code, out, _ = run(capsys, "analyze", sample_session_path, "--json", "--no-turns")
    assert code == 0
    assert "turns" not in json.loads(out)


def test_analyze_markdown(capsys, sample_session_path):
    code, out, _ = run(capsys, "analyze", sample_session_path, "--markdown")
    assert code == 0
    assert out.startswith("# Fix failing test in app.py")
    assert "| Cache read (carrying context) |" in out
    assert "## Findings" in out


def test_analyze_csv(capsys, sample_session_path, tmp_path):
    target = tmp_path / "turns.csv"
    code, _, err = run(capsys, "analyze", sample_session_path, "--csv", target)
    assert code == 0
    assert "wrote 6 turns" in err
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[0]["model"] == "claude-opus-5"
    assert rows[0]["cache_write_1h_tokens"] == "12000"
    assert rows[0]["tool_names"] == "Read"
    assert rows[-1]["model"] == "claude-sonnet-5"


def test_analyze_missing_target(capsys, projects_dir):
    code, out, err = run(capsys, "--projects-dir", projects_dir, "analyze", "nope-nope")
    assert code == 1
    assert "tokenlens:" in err and "unique session ID prefix" in err


def test_analyze_rejects_empty_transcript(capsys, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"type": "queue-operation"}\n', encoding="utf-8")
    code, _, err = run(capsys, "analyze", empty)
    assert code == 1 and "no billed API calls" in err


def test_sessions_table_and_json(capsys, projects_dir):
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "sessions")
    assert code == 0
    assert "1 sessions, 6 API calls" in out
    assert SESSION_ID[:8] in out
    assert "Fix failing test in app.py" in out

    code, out, _ = run(capsys, "--projects-dir", projects_dir, "sessions", "--json")
    data = json.loads(out)
    assert data[0]["session_id"] == SESSION_ID
    assert data[0]["turns"] == 6


def test_sessions_project_filter(capsys, projects_dir):
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "sessions", "--project", "demo-app")
    assert code == 0 and SESSION_ID[:8] in out
    code, _, err = run(capsys, "--projects-dir", projects_dir, "sessions", "--project", "zzz")
    assert code == 1 and "no sessions found" in err


def test_summary(capsys, projects_dir):
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "summary", "--daily")
    assert code == 0
    assert "Claude Code usage summary" in out
    assert "By project" in out and "By model" in out
    assert "Top findings across sessions" in out
    assert "Daily API-equivalent cost" in out
    assert "2026-09-01" in out

    code, out, _ = run(capsys, "--projects-dir", projects_dir, "summary", "--json")
    data = json.loads(out)
    assert data["session_count"] == 1
    assert data["by_project"][0]["project_path"] == "C:/Users/Someone/Projects/demo-app"
    assert data["daily"][0]["day"] == "2026-09-01"


def test_findings_command(capsys, projects_dir):
    code, out, _ = run(capsys, "--projects-dir", projects_dir, "findings")
    assert code == 0
    assert "findings across 1 sessions" in out
    assert "[WARN] Re-read an unchanged file" in out
    assert f"/ {SESSION_ID[:8]}]" in out  # provenance chip

    code, out, _ = run(capsys, "--projects-dir", projects_dir, "findings", "--json", "--min-dollars", "1000")
    assert code == 0 and json.loads(out) == []


def test_models_command(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    assert "claude-opus-5" in out and "claude-haiku-4-5" in out
    assert "Write 1h" in out

    code, out, _ = run(capsys, "models", "--json")
    data = {m["key"]: m for m in json.loads(out)}
    assert data["claude-opus-5"]["rates"]["cache_write_1h"] == 10.0
    assert data["claude-opus-5"]["fast_rates"]["input"] == 10.0


def test_pricing_override_flag(capsys, sample_session_path, tmp_path):
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps({"claude-opus-5": {"input": 0, "output": 0,
                                                    "cache_read_multiplier": 0, "cache_write_1h_multiplier": 0}}),
                     encoding="utf-8")
    code, out, _ = run(capsys, "--pricing", rates, "analyze", sample_session_path, "--json")
    data = json.loads(out)
    opus = next(m for m in data["by_model"] if m["model"] == "claude-opus-5")
    assert opus["cost"]["total"] == 0.0


def test_bad_pricing_file(capsys, sample_session_path, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    code, _, err = run(capsys, "--pricing", bad, "analyze", sample_session_path)
    assert code == 1 and "pricing overrides" in err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("tokenlens ")
