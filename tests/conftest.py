"""Builders for synthetic Claude Code transcript lines.

Real transcripts contain private data, so tests construct their own from
these helpers. The shapes mirror what Claude Code 2.x actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

SESSION_ID = "11111111-2222-3333-4444-555555555555"
CWD = "C:\\Users\\Someone\\Projects\\demo-app"


def _base(entry_type: str, ts: str, **extra: Any) -> Dict[str, Any]:
    entry = {
        "type": entry_type,
        "timestamp": ts,
        "sessionId": SESSION_ID,
        "cwd": CWD,
        "version": "2.1.263",
        "gitBranch": "main",
        "entrypoint": "cli",
        "isSidechain": False,
        "uuid": f"uuid-{ts}-{entry_type}",
    }
    entry.update(extra)
    return entry


def usage(input_tokens=0, output_tokens=0, cache_read=0, cache_5m=0, cache_1h=0,
          speed="standard", split=True) -> Dict[str, Any]:
    u: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_5m + cache_1h,
        "service_tier": "standard",
        "speed": speed,
    }
    if split:
        u["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_5m,
            "ephemeral_1h_input_tokens": cache_1h,
        }
    return u


def assistant_line(message_id: str, ts: str, block: Dict[str, Any], usage_dict: Dict[str, Any],
                   model="claude-opus-5", effort="high", stop_reason=None) -> Dict[str, Any]:
    return _base(
        "assistant", ts,
        requestId=f"req_{message_id}",
        effort=effort,
        message={
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [block],
            "stop_reason": stop_reason,
            "usage": usage_dict,
        },
    )


def text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def thinking_block(text: str) -> Dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": "sig"}


def tool_use_block(tool_id: str, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def user_prompt_line(ts: str, text: str) -> Dict[str, Any]:
    return _base("user", ts, message={"role": "user", "content": text})


def tool_result_line(ts: str, tool_id: str, content: str, is_error: bool = False) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_id, "content": content}
    if is_error:
        block["is_error"] = True
    return _base("user", ts, message={"role": "user", "content": [block]},
                 toolUseResult={"type": "text"})


def compaction_line(ts: str, pre: int, post: int, trigger="auto") -> Dict[str, Any]:
    return _base(
        "system", ts,
        subtype="compact_boundary",
        content="Conversation compacted",
        level="info",
        compactMetadata={"trigger": trigger, "preTokens": pre, "postTokens": post},
    )


def api_error_line(ts: str, status: int = 429, error: str = "rate_limit") -> Dict[str, Any]:
    return _base(
        "assistant", ts,
        isApiErrorMessage=True,
        apiErrorStatus=status,
        error=error,
        message={
            "id": "local-error-id",
            "role": "assistant",
            "model": "<synthetic>",
            "content": [{"type": "text", "text": "You've hit your session limit"}],
            "usage": usage(),
        },
    )


def title_line(title: str) -> Dict[str, Any]:
    return {"type": "ai-title", "aiTitle": title, "sessionId": SESSION_ID}


def write_jsonl(path: Path, entries: List[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry if isinstance(entry, str) else json.dumps(entry))
            handle.write("\n")
    return path


def build_sample_session() -> List[Dict[str, Any]]:
    """A small but realistic session used across the test suite.

    Timeline (all on 2026-09-01):
      turn 1  msg_a  text + tool_use Read(app.py)        1h cache write 12,000
      turn 2  msg_b  tool_use Bash(pytest)                cache read 12,000, 1h write 3,000
      turn 3  msg_c  tool_use Read(app.py)  <- duplicate  cache read 15,000, write 1,000
      turn 4  msg_d  tool_use Edit(app.py)                cache read 16,000, write 500
      turn 5  msg_e  tool_use Read(app.py)  <- legit, file was edited in turn 4
      compaction (pre 40,000 -> post 5,000)
      api error 429
      turn 6  msg_f  text only, sonnet-5                  cache read 5,000
    """
    big_result = "x" * 30_000  # ~7,500 tokens of file content
    return [
        user_prompt_line("2026-09-01T10:00:00.000Z", "Please fix the failing test in app.py"),
        assistant_line("msg_a", "2026-09-01T10:00:05.000Z", text_block("Let me look at the file first."),
                       usage(input_tokens=10, output_tokens=120, cache_1h=12_000)),
        assistant_line("msg_a", "2026-09-01T10:00:06.000Z",
                       tool_use_block("toolu_1", "Read", {"file_path": "C:/proj/app.py"}),
                       usage(input_tokens=10, output_tokens=160, cache_1h=12_000), stop_reason="tool_use"),
        tool_result_line("2026-09-01T10:00:07.000Z", "toolu_1", big_result),
        assistant_line("msg_b", "2026-09-01T10:00:20.000Z",
                       tool_use_block("toolu_2", "Bash", {"command": "pytest -q", "description": "Run tests"}),
                       usage(input_tokens=5, output_tokens=60, cache_read=12_000, cache_1h=3_000), stop_reason="tool_use"),
        tool_result_line("2026-09-01T10:00:30.000Z", "toolu_2", "1 failed, 3 passed", is_error=True),
        assistant_line("msg_c", "2026-09-01T10:00:40.000Z",
                       tool_use_block("toolu_3", "Read", {"file_path": "C:/proj/app.py"}),
                       usage(input_tokens=5, output_tokens=40, cache_read=15_000, cache_1h=1_000), stop_reason="tool_use"),
        tool_result_line("2026-09-01T10:00:41.000Z", "toolu_3", big_result),
        assistant_line("msg_d", "2026-09-01T10:01:00.000Z",
                       tool_use_block("toolu_4", "Edit", {"file_path": "C:/proj/app.py", "old_string": "a", "new_string": "b"}),
                       usage(input_tokens=5, output_tokens=80, cache_read=16_000, cache_1h=500), stop_reason="tool_use"),
        tool_result_line("2026-09-01T10:01:01.000Z", "toolu_4", "The file has been updated."),
        assistant_line("msg_e", "2026-09-01T10:01:10.000Z",
                       tool_use_block("toolu_5", "Read", {"file_path": "C:/proj/app.py"}),
                       usage(input_tokens=5, output_tokens=40, cache_read=16_500, cache_1h=200), stop_reason="tool_use"),
        tool_result_line("2026-09-01T10:01:11.000Z", "toolu_5", big_result),
        compaction_line("2026-09-01T10:02:00.000Z", pre=40_000, post=5_000),
        api_error_line("2026-09-01T10:02:05.000Z"),
        title_line("Fix failing test in app.py"),
        assistant_line("msg_f", "2026-09-01T10:03:00.000Z", text_block("Done. The test passes now."),
                       usage(input_tokens=20, output_tokens=200, cache_read=5_000), model="claude-sonnet-5",
                       stop_reason="end_turn"),
    ]


@pytest.fixture
def sample_entries() -> List[Dict[str, Any]]:
    return build_sample_session()


@pytest.fixture
def sample_session_path(tmp_path: Path) -> Path:
    project_dir = tmp_path / "projects" / "C--Users-Someone-Projects-demo-app"
    return write_jsonl(project_dir / f"{SESSION_ID}.jsonl", build_sample_session())
