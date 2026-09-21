"""Turn a Claude Code JSONL transcript into a :class:`~tokenlens.models.Session`.

Transcript format, as observed in Claude Code 2.x:

- Every line is a JSON object with a ``type``. The ones that matter for
  cost are ``assistant`` (API responses, carrying ``message.usage``),
  ``user`` (human prompts and ``tool_result`` blocks), and ``system``
  lines with ``subtype == "compact_boundary"`` (context compaction).
  Other types (``attachment``, ``ai-title``, ``file-history-snapshot``,
  ``queue-operation``...) are metadata.
- An assistant response is split across several lines, one per content
  block, all sharing ``message.id`` and repeating the same ``usage``.
  One ``message.id`` is one billed API call, so lines are folded back
  together by ID.
- Assistant lines with ``isApiErrorMessage: true`` are locally generated
  (rate limits, 5xx). They carry ``model: "<synthetic>"`` and zero usage
  and are recorded as :class:`~tokenlens.models.ApiError`, not turns.
- Every line records ``cwd``, which is the exact project directory. The
  directory name under ``~/.claude/projects`` is a lossy encoding of it
  (``:``, ``/``, ``\\``, spaces, and ``_`` all become ``-``), so ``cwd``
  is always preferred.
- Subagent transcripts live at
  ``<project-dir>/<session-id>/subagents/agent-<id>.jsonl``, or one level
  deeper at ``.../subagents/workflows/wf_<id>/agent-<id>.jsonl`` when the
  Workflow tool spawned them. They carry ``isSidechain: true`` plus an
  ``agentId``. Their cost is real and is attached to the parent session;
  on a machine that uses workflows they can outweigh the main sessions.

Files are read as UTF-8 with replacement so a stray byte never aborts an
analysis, and malformed lines are counted and skipped rather than raised.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import ApiError, CompactionEvent, Session, TokenUsage, ToolCall, Turn
from .pricing import DEFAULT_TABLE, PricingTable

PREVIEW_CHARS = 160
TARGET_CHARS = 120
FIRST_PROMPT_CHARS = 200

_DRIVE_LETTER = re.compile(r"^[a-zA-Z]:")


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse the ISO-8601 timestamps Claude Code writes (``...Z`` suffix).

    Returns an aware UTC datetime, or ``None`` when the value is missing
    or unparseable.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_project_path(cwd: str) -> str:
    """Make ``cwd`` values comparable across sessions.

    Windows transcripts alternate between ``c:\\Users\\...`` and
    ``C:\\Users\\...`` depending on how Claude Code was launched. Forward
    slashes and an upper-case drive letter give one canonical spelling
    without changing the path's meaning.
    """
    if not cwd:
        return ""
    path = cwd.replace("\\", "/")
    if _DRIVE_LETTER.match(path):
        path = path[0].upper() + path[1:]
    return path.rstrip("/") or path


def _usage_from_dict(usage: Dict[str, Any]) -> TokenUsage:
    def as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    cache_total = as_int(usage.get("cache_creation_input_tokens"))
    creation = usage.get("cache_creation")
    if isinstance(creation, dict):
        five_min = as_int(creation.get("ephemeral_5m_input_tokens"))
        one_hour = as_int(creation.get("ephemeral_1h_input_tokens"))
        if five_min + one_hour == 0 and cache_total:
            # Sub-object present but empty: fall back to the total.
            five_min = cache_total
    else:
        # Older transcripts have no TTL split. The API default TTL is five
        # minutes, so that is the conservative assumption.
        five_min, one_hour = cache_total, 0

    return TokenUsage(
        input_tokens=as_int(usage.get("input_tokens")),
        output_tokens=as_int(usage.get("output_tokens")),
        cache_read_tokens=as_int(usage.get("cache_read_input_tokens")),
        cache_write_5m_tokens=five_min,
        cache_write_1h_tokens=one_hour,
    )


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                total += len(block.get("text") or "")
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


def _text_preview(content: Any, limit: int = PREVIEW_CHARS) -> str:
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return (block.get("text") or "")[:limit]
            if isinstance(block, str):
                return block[:limit]
    return ""


def _first_line(text: str, limit: int = TARGET_CHARS) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


def tool_target(name: str, tool_input: Any) -> str:
    """Short description of what a tool call was aimed at."""
    if not isinstance(tool_input, dict):
        return ""
    if name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if name == "Read" and (tool_input.get("offset") or tool_input.get("limit")):
            target += f" [offset={tool_input.get('offset', 0)} limit={tool_input.get('limit', '-')}]"
        return target
    if name in ("Bash", "PowerShell"):
        return _first_line(str(tool_input.get("command") or ""))
    if name in ("Grep", "Glob"):
        pattern = str(tool_input.get("pattern") or "")
        path = str(tool_input.get("path") or "")
        return f"{pattern} in {path}" if path else pattern
    if name == "WebFetch":
        return str(tool_input.get("url") or "")
    if name == "WebSearch":
        return str(tool_input.get("query") or "")
    if name == "Agent":
        return f"{tool_input.get('subagent_type', 'agent')}: {tool_input.get('description', '')}".strip(": ")
    if name == "Skill":
        return str(tool_input.get("skill") or "")
    if name == "TodoWrite":
        todos = tool_input.get("todos")
        return f"{len(todos)} items" if isinstance(todos, list) else ""
    if "description" in tool_input:
        return _first_line(str(tool_input["description"]))
    try:
        return json.dumps(tool_input, ensure_ascii=False)[:TARGET_CHARS]
    except (TypeError, ValueError):
        return ""


def _is_human_prompt(entry: Dict[str, Any]) -> bool:
    """True for a user line typed by the human, not a tool_result carrier."""
    if entry.get("isMeta") or entry.get("isCompactSummary"):
        return False
    message = entry.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_text = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                return False
            if block.get("type") == "text" and (block.get("text") or "").strip():
                has_text = True
        return has_text
    return False


def iter_json_lines(path: Path) -> Iterable[Any]:
    """Yield parsed objects; yield ``None`` for lines that fail to parse."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                yield None


class _Builder:
    """Accumulates state across one pass over a transcript."""

    def __init__(self, path: str, pricing: PricingTable):
        self.pricing = pricing
        self.session = Session(path=path, session_id=Path(path).stem)
        self.turns_by_id: Dict[str, Turn] = {}
        self.order: List[str] = []
        self.tool_results: Dict[str, Dict[str, Any]] = {}
        self.pending_calls: List[ToolCall] = []
        self.speed_by_id: Dict[str, str] = {}

    # -- per-line dispatch ---------------------------------------------

    def feed(self, entry: Any) -> None:
        if entry is None:
            self.session.skipped_lines += 1
            return
        if not isinstance(entry, dict):
            self.session.skipped_lines += 1
            return

        self._capture_metadata(entry)
        kind = entry.get("type")
        if kind == "assistant":
            self._assistant(entry)
        elif kind == "user":
            self._user(entry)
        elif kind == "system":
            self._system(entry)
        elif kind == "ai-title":
            title = entry.get("aiTitle")
            if isinstance(title, str) and title.strip():
                self.session.title = title.strip()

    def _capture_metadata(self, entry: Dict[str, Any]) -> None:
        s = self.session
        if not s.project_path and entry.get("cwd"):
            s.project_path = normalize_project_path(str(entry["cwd"]))
        if entry.get("sessionId") and not s.parent_session_id and entry.get("isSidechain"):
            s.parent_session_id = str(entry["sessionId"])
        if entry.get("agentId") and not s.agent_id:
            s.agent_id = str(entry["agentId"])
            s.is_subagent = True
        if not s.git_branch and entry.get("gitBranch"):
            s.git_branch = str(entry["gitBranch"])
        if not s.entrypoint and entry.get("entrypoint"):
            s.entrypoint = str(entry["entrypoint"])
        version = entry.get("version")
        if isinstance(version, str) and version and version not in s.claude_versions:
            s.claude_versions.append(version)

    def _assistant(self, entry: Dict[str, Any]) -> None:
        message = entry.get("message") or {}
        timestamp = parse_timestamp(entry.get("timestamp"))

        if entry.get("isApiErrorMessage"):
            status = entry.get("apiErrorStatus")
            self.session.api_errors.append(ApiError(
                timestamp=timestamp,
                status=int(status) if isinstance(status, int) else None,
                error=str(entry.get("error") or ""),
                message=_text_preview(message.get("content")),
                after_turn=len(self.order),
            ))
            return

        message_id = message.get("id")
        usage = message.get("usage")
        if not message_id or not isinstance(usage, dict):
            return

        turn = self.turns_by_id.get(message_id)
        if turn is None:
            turn = Turn(index=len(self.order) + 1, message_id=str(message_id), timestamp=timestamp)
            self.turns_by_id[message_id] = turn
            self.order.append(message_id)

        # Later lines of the same message carry the final numbers.
        turn.usage = _usage_from_dict(usage)
        turn.model = message.get("model") or turn.model
        turn.speed = str(usage.get("speed") or turn.speed or "standard")
        turn.effort = str(entry.get("effort") or turn.effort)
        turn.request_id = str(entry.get("requestId") or turn.request_id)
        if message.get("stop_reason"):
            turn.stop_reason = str(message["stop_reason"])
        if timestamp and (turn.timestamp is None or timestamp < turn.timestamp):
            turn.timestamp = timestamp

        content = message.get("content")
        if isinstance(content, str):
            turn.text_chars += len(content)
            return
        for block in content or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                turn.thinking_chars += len(block.get("thinking") or "")
            elif block_type == "text":
                turn.text_chars += len(block.get("text") or "")
            elif block_type == "tool_use":
                block_id = str(block.get("id") or "")
                if block_id and any(c.id == block_id for c in turn.tool_calls):
                    continue  # same block written on a later line of this message
                tool_input = block.get("input")
                try:
                    input_chars = len(json.dumps(tool_input, ensure_ascii=False)) if tool_input else 0
                except (TypeError, ValueError):
                    input_chars = 0
                call = ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or "unknown"),
                    target=tool_target(str(block.get("name") or ""), tool_input),
                    input_chars=input_chars,
                )
                turn.tool_calls.append(call)
                self.pending_calls.append(call)

    def _user(self, entry: Dict[str, Any]) -> None:
        message = entry.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if tool_id:
                        self.tool_results[str(tool_id)] = block
        if _is_human_prompt(entry):
            self.session.user_prompt_count += 1
            if not self.session.first_prompt:
                self.session.first_prompt = _text_preview(content, FIRST_PROMPT_CHARS).strip()

    def _system(self, entry: Dict[str, Any]) -> None:
        if entry.get("subtype") != "compact_boundary":
            return
        meta = entry.get("compactMetadata") or {}
        self.session.compactions.append(CompactionEvent(
            timestamp=parse_timestamp(entry.get("timestamp")),
            trigger=str(meta.get("trigger") or ""),
            pre_tokens=int(meta.get("preTokens") or 0),
            post_tokens=int(meta.get("postTokens") or 0),
            after_turn=len(self.order),
        ))

    # -- finalization --------------------------------------------------

    def finish(self) -> Session:
        for call in self.pending_calls:
            result = self.tool_results.get(call.id)
            if result is None:
                continue
            call.result_chars = _text_len(result.get("content"))
            call.result_preview = _text_preview(result.get("content"))
            call.is_error = bool(result.get("is_error"))

        turns = [self.turns_by_id[mid] for mid in self.order]
        for turn in turns:
            resolution = self.pricing.resolve(turn.model, turn.speed)
            turn.cost = self.pricing.price_with_rates(turn.usage, resolution.rates)
            turn.pricing_is_estimate = resolution.is_estimate
        self.session.turns = turns
        return self.session


def parse_entries(entries: Iterable[Any], path: str = "<memory>",
                  pricing: PricingTable = DEFAULT_TABLE) -> Session:
    """Build a Session from already-decoded JSON objects (mainly for tests)."""
    builder = _Builder(path, pricing)
    for entry in entries:
        builder.feed(entry)
    return builder.finish()


def subagent_dir_for(path: Path) -> Path:
    return path.with_suffix("") / "subagents"


def workflow_id_for(sub_path: Path, sub_dir: Path) -> str:
    """Workflow-spawned agents sit under ``subagents/workflows/wf_<id>/``."""
    try:
        parts = sub_path.relative_to(sub_dir).parts
    except ValueError:
        return ""
    for part in parts[:-1]:
        if part.startswith("wf_"):
            return part
    return ""


def parse_session(path: str, pricing: PricingTable = DEFAULT_TABLE,
                  include_subagents: bool = True) -> Session:
    """Parse one transcript file, attaching its subagent transcripts."""
    file_path = Path(path)
    builder = _Builder(str(file_path), pricing)
    for entry in iter_json_lines(file_path):
        builder.feed(entry)
    session = builder.finish()
    session.project_dir = file_path.parent.name

    if include_subagents and not session.is_subagent:
        sub_dir = subagent_dir_for(file_path)
        if sub_dir.is_dir():
            for sub_path in sorted(sub_dir.rglob("*.jsonl")):
                sub = parse_session(str(sub_path), pricing, include_subagents=False)
                sub.is_subagent = True
                sub.workflow_id = workflow_id_for(sub_path, sub_dir)
                sub.parent_session_id = sub.parent_session_id or session.session_id
                if not sub.agent_id:
                    stem = sub_path.stem
                    sub.agent_id = stem[len("agent-"):] if stem.startswith("agent-") else stem
                sub.project_path = sub.project_path or session.project_path
                sub.project_dir = session.project_dir
                session.subagents.append(sub)
    return session
