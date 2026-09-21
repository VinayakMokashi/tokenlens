"""Export reports as JSON, CSV, or Markdown.

JSON is the machine-readable twin of the text report and is what the
web dashboard consumes. CSV gives one row per turn for spreadsheets.
Markdown is a shareable write-up of a single session.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .analysis import AggregateReport, SessionReport
from .findings import Finding, total_estimated_savings
from .formatting import money, pct, signed_money, tokens, when
from .models import Session, Turn

SCHEMA_VERSION = 1


def jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, datetimes, and paths to JSON types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _usage_dict(usage) -> Dict[str, int]:
    d = jsonable(usage)
    d["cache_write_tokens"] = usage.cache_write_tokens
    d["context_tokens"] = usage.context_tokens
    d["total_tokens"] = usage.total_tokens
    return d


def _cost_dict(cost) -> Dict[str, float]:
    d = jsonable(cost)
    d["cache_write_cost"] = cost.cache_write_cost
    d["total"] = cost.total
    d["cache_savings"] = cost.cache_savings
    return d


def turn_to_dict(turn: Turn) -> Dict[str, Any]:
    return {
        "index": turn.index,
        "message_id": turn.message_id,
        "timestamp": jsonable(turn.timestamp),
        "model": turn.model,
        "speed": turn.speed,
        "effort": turn.effort,
        "stop_reason": turn.stop_reason,
        "kind": turn.kind,
        "usage": _usage_dict(turn.usage),
        "cost": _cost_dict(turn.cost),
        "thinking_chars": turn.thinking_chars,
        "text_chars": turn.text_chars,
        "tool_calls": jsonable(turn.tool_calls),
        "pricing_is_estimate": turn.pricing_is_estimate,
    }


def session_meta(session: Session) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "path": session.path,
        "project_path": session.project_path,
        "project_dir": session.project_dir,
        "title": session.title,
        "first_prompt": session.first_prompt,
        "git_branch": session.git_branch,
        "entrypoint": session.entrypoint,
        "claude_versions": session.claude_versions,
        "models": session.models,
        "started_at": jsonable(session.started_at),
        "ended_at": jsonable(session.ended_at),
        "wall_seconds": jsonable(session.wall_duration),
        "active_seconds": session.active_duration().total_seconds(),
        "user_prompt_count": session.user_prompt_count,
        "turn_count": len(session.turns),
        "is_subagent": session.is_subagent,
        "agent_id": session.agent_id,
        "workflow_id": session.workflow_id,
        "parent_session_id": session.parent_session_id,
        "skipped_lines": session.skipped_lines,
    }


def session_report_to_dict(report: SessionReport, findings: Sequence[Finding],
                           include_turns: bool = True) -> Dict[str, Any]:
    s = report.session
    data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session": session_meta(s),
        "usage": _usage_dict(report.usage),
        "cost": _cost_dict(report.cost),
        "usage_with_subagents": _usage_dict(report.usage_with_subagents),
        "cost_with_subagents": _cost_dict(report.cost_with_subagents),
        "metrics": {
            "peak_context": report.peak_context,
            "mean_context": report.mean_context,
            "carry_cost": report.carry_cost,
            "new_work_cost": report.new_work_cost,
            "carry_share": report.carry_share,
            "cache_hit_rate": report.cache_hit_rate,
            "cost_per_prompt": report.cost_per_prompt,
            "cost_per_turn": report.cost_per_turn,
            "subagent_cost": s.subagent_cost,
            "has_estimated_pricing": report.has_estimated_pricing,
            "estimated_avoidable": total_estimated_savings(findings),
        },
        "by_model": [{**jsonable(m), "usage": _usage_dict(m.usage), "cost": _cost_dict(m.cost)} for m in report.by_model],
        "by_kind": {k: {"turns": v[0], "cost": v[1]} for k, v in report.by_kind.items()},
        "tools": [{**jsonable(t), "est_result_tokens": t.est_result_tokens} for t in report.tools],
        "context": jsonable(report.context),
        "phases": jsonable(report.phases),
        "compactions": jsonable(s.compactions),
        "api_errors": jsonable(s.api_errors),
        "what_if": [{**jsonable(w), "delta_pct": w.delta_pct} for w in report.what_if],
        "daily": jsonable(report.daily),
        "findings": jsonable(findings),
        "subagents": [
            {**session_meta(sub), "cost": sub.total_cost, "usage": _usage_dict(sub.usage)}
            for sub in s.subagents
        ],
    }
    if include_turns:
        data["turns"] = [turn_to_dict(t) for t in s.turns]
    return data


def aggregate_to_dict(report: AggregateReport, findings: Sequence[Finding]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_count": report.session_count,
        "turn_count": report.turn_count,
        "subagent_turn_count": report.subagent_turn_count,
        "total_turn_count": report.total_turn_count,
        "total_cost": report.total_cost,
        "total_subagent_cost": report.total_subagent_cost,
        "usage": _usage_dict(report.usage),
        "cost": _cost_dict(report.cost),
        "has_estimated_pricing": report.has_estimated_pricing,
        "by_project": jsonable(report.by_project),
        "by_model": [{**jsonable(m), "usage": _usage_dict(m.usage), "cost": _cost_dict(m.cost)} for m in report.by_model],
        "daily": jsonable(report.daily),
        "tools": [{**jsonable(t), "est_result_tokens": t.est_result_tokens} for t in report.tools],
        "what_if": [{**jsonable(w), "delta_pct": w.delta_pct} for w in report.what_if],
        "sessions": jsonable(report.sessions),
        "findings": jsonable(findings),
        "estimated_avoidable": total_estimated_savings(findings),
    }


def to_json(data: Any, indent: Optional[int] = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


TURN_CSV_COLUMNS = [
    "index", "timestamp", "model", "speed", "effort", "kind", "stop_reason",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_5m_tokens",
    "cache_write_1h_tokens", "context_tokens", "cost_total", "cost_input", "cost_output",
    "cost_cache_read", "cost_cache_write", "thinking_chars", "text_chars",
    "tool_calls", "tool_names", "tool_result_chars", "tool_errors", "pricing_is_estimate",
]


def write_turns_csv(session: Session, path: str) -> int:
    """Write one row per turn; returns the number of rows written."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TURN_CSV_COLUMNS)
        for t in session.turns:
            writer.writerow([
                t.index, jsonable(t.timestamp) or "", t.model, t.speed, t.effort, t.kind, t.stop_reason,
                t.usage.input_tokens, t.usage.output_tokens, t.usage.cache_read_tokens,
                t.usage.cache_write_5m_tokens, t.usage.cache_write_1h_tokens, t.usage.context_tokens,
                f"{t.cost.total:.6f}", f"{t.cost.input_cost:.6f}", f"{t.cost.output_cost:.6f}",
                f"{t.cost.cache_read_cost:.6f}", f"{t.cost.cache_write_cost:.6f}",
                t.thinking_chars, t.text_chars, len(t.tool_calls),
                ";".join(c.name for c in t.tool_calls),
                sum(c.result_chars or 0 for c in t.tool_calls),
                sum(1 for c in t.tool_calls if c.is_error),
                t.pricing_is_estimate,
            ])
    return len(session.turns)


def render_markdown(report: SessionReport, findings: Sequence[Finding], limit: int = 15) -> str:
    s = report.session
    c = report.cost
    u = report.usage
    total = c.total or 1.0
    lines: List[str] = []
    lines.append(f"# {s.title or s.first_prompt[:70] or 'Claude Code session'}")
    lines.append("")
    lines.append(f"- **Project:** `{s.project_path or s.project_dir}`")
    lines.append(f"- **Session:** `{s.session_id}`")
    lines.append(f"- **When:** {when(s.started_at)} to {when(s.ended_at)}")
    lines.append(f"- **Turns:** {report.turn_count} API calls for {s.user_prompt_count} prompts, "
                 f"{len(s.subagents)} subagents, {len(s.compactions)} compactions")
    lines.append(f"- **Models:** {', '.join(s.models) or '-'}")
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"**{money(c.total)}** for this conversation" +
                 (f", **{money(report.cost_with_subagents.total)}** including subagents." if s.subagents else "."))
    lines.append("")
    lines.append("| Category | Tokens | Cost | Share |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Cache read (carrying context) | {tokens(u.cache_read_tokens)} | {money(c.cache_read_cost)} | {pct(c.cache_read_cost / total)} |")
    lines.append(f"| Cache write | {tokens(u.cache_write_tokens)} | {money(c.cache_write_cost)} | {pct(c.cache_write_cost / total)} |")
    lines.append(f"| Fresh input | {tokens(u.input_tokens)} | {money(c.input_cost)} | {pct(c.input_cost / total)} |")
    lines.append(f"| Output (incl. thinking) | {tokens(u.output_tokens)} | {money(c.output_cost)} | {pct(c.output_cost / total)} |")
    lines.append("")
    lines.append(f"Carrying existing context cost {money(report.carry_cost)} ({pct(report.carry_share)}); "
                 f"new work cost {money(report.new_work_cost)}. Cache hit rate {pct(report.cache_hit_rate, 1)}; "
                 f"peak context {tokens(report.peak_context)} tokens.")
    lines.append("")
    if report.what_if:
        lines.append("## Same conversation on another model")
        lines.append("")
        lines.append("| Model | Cost | vs. actual |")
        lines.append("|---|---:|---:|")
        for w in report.what_if:
            delta = f"{signed_money(w.delta)} ({w.delta_pct:+.0f}%)" if w.delta_pct is not None else signed_money(w.delta)
            lines.append(f"| {w.display} | {money(w.cost)} | {delta} |")
        lines.append("")
        lines.append("_First-order estimate: token counts held fixed; tokenizers and turn counts differ across models._")
        lines.append("")
    lines.append("## Findings")
    lines.append("")
    savings = total_estimated_savings(findings)
    if savings >= 0.005:
        lines.append(f"Estimated avoidable spend: **~{money(savings)}** of {money(c.total)}.")
        lines.append("")
    if not findings:
        lines.append("No notable inefficiencies detected.")
    for f in list(findings)[:limit]:
        save = f" (~{money(f.est_dollars_saved)} avoidable)" if f.est_dollars_saved >= 0.005 else ""
        lines.append(f"- **{f.severity.upper()}: {f.title}**{save}  ")
        lines.append(f"  {f.detail}")
    lines.append("")
    lines.append("_Costs are API-equivalent estimates from recorded token counts at public rates._")
    return "\n".join(lines)
