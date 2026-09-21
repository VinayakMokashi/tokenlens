"""Plain-text rendering of session and aggregate reports for the CLI.

Output is ASCII only so it renders correctly on Windows consoles with
legacy code pages and in CI logs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .analysis import AggregateReport, SessionReport
from .findings import Finding, total_estimated_savings
from .formatting import duration, heading, money, pct, shorten, signed_money, table, tokens, when, wrap

WIDTH = 78

DISCLAIMER = (
    "Costs are API-equivalent estimates from recorded token counts at public "
    "per-token rates. Subscription (Pro/Max) users pay a flat fee; the figures "
    "show what the same usage would cost on the API and how it compares across "
    "sessions."
)


def _severity_tag(severity: str) -> str:
    return {"critical": "[CRIT]", "warning": "[WARN]", "info": "[info]"}.get(severity, "[    ]")


def render_findings(findings: Sequence[Finding], limit: Optional[int] = 10,
                    show_provenance: bool = False) -> str:
    if not findings:
        return "No notable inefficiencies detected."
    lines: List[str] = []
    shown = list(findings) if limit is None else list(findings)[:limit]
    for i, f in enumerate(shown, start=1):
        savings = f"  (~{money(f.est_dollars_saved)} avoidable)" if f.est_dollars_saved >= 0.005 else ""
        where = ""
        if show_provenance and f.project_path:
            where = f"  [{shorten(f.project_path.rstrip('/').split('/')[-1], 28)} / {f.session_id[:8]}]"
        lines.append(f"{i:>2}. {_severity_tag(f.severity)} {f.title}{savings}{where}")
        lines.append(wrap(f.detail, WIDTH, indent="      "))
        lines.append("")
    hidden = len(findings) - len(shown)
    if hidden > 0:
        lines.append(f"      ... {hidden} more finding{'s' if hidden != 1 else ''} (use --all-findings to show every one)")
    return "\n".join(lines).rstrip()


def render_session(report: SessionReport, findings: Sequence[Finding],
                   findings_limit: Optional[int] = 10) -> str:
    s = report.session
    out: List[str] = []

    title = s.title or shorten(s.first_prompt, 70) or "(untitled session)"
    out.append(heading(f"Session: {title}", WIDTH))
    out.append(f"Project:   {s.project_path or s.project_dir or '-'}")
    out.append(f"Session:   {s.session_id}")
    out.append(f"When:      {when(s.started_at)} -> {when(s.ended_at)}"
               f"   (active {duration(s.active_duration())}, wall {duration(s.wall_duration)})")
    versions = ", ".join(s.claude_versions[:3]) + (" ..." if len(s.claude_versions) > 3 else "")
    out.append(f"Claude:    {versions or '-'}   models: {', '.join(s.models) or '-'}"
               f"{'   branch: ' + s.git_branch if s.git_branch else ''}")
    out.append(f"Turns:     {report.turn_count} API calls for {s.user_prompt_count} prompts"
               f"   subagents: {len(s.subagents)}   compactions: {len(s.compactions)}"
               f"   API errors: {len(s.api_errors)}")
    if s.skipped_lines:
        out.append(f"Note:      {s.skipped_lines} unreadable line(s) skipped")
    out.append("")

    # -- cost --------------------------------------------------------------
    c = report.cost
    total = c.total
    out.append(heading("Cost", WIDTH, "-"))
    out.append(f"Total (this conversation):  {money(total)}")
    if s.subagents:
        out.append(f"Subagents ({len(s.subagents)}):             {money(s.subagent_cost)}")
        out.append(f"Total including subagents:  {money(report.cost_with_subagents.total)}")
    out.append("")
    u = report.usage

    def share(v: float) -> str:
        return pct(v / total if total else 0.0)

    out.append(table(
        ["Category", "Tokens", "Cost", "Share"],
        [
            ["Cache read (carrying context)", tokens(u.cache_read_tokens), money(c.cache_read_cost), share(c.cache_read_cost)],
            ["Cache write (1h)", tokens(u.cache_write_1h_tokens),
             money(u.cache_write_1h_tokens and c.cache_write_cost * u.cache_write_1h_tokens / max(u.cache_write_tokens, 1)), ""],
            ["Cache write (5m)", tokens(u.cache_write_5m_tokens),
             money(u.cache_write_5m_tokens and c.cache_write_cost * u.cache_write_5m_tokens / max(u.cache_write_tokens, 1)), ""],
            ["Cache write (all)", tokens(u.cache_write_tokens), money(c.cache_write_cost), share(c.cache_write_cost)],
            ["Fresh input", tokens(u.input_tokens), money(c.input_cost), share(c.input_cost)],
            ["Output (incl. thinking)", tokens(u.output_tokens), money(c.output_cost), share(c.output_cost)],
        ],
        align=["l", "r", "r", "r"],
    ))
    out.append("")
    out.append(f"Carrying existing context:  {money(report.carry_cost)} ({pct(report.carry_share)})"
               f"    New work:  {money(report.new_work_cost)}")
    out.append(f"Cache hit rate:             {pct(report.cache_hit_rate, 1)} of prompt tokens served from cache;"
               f" cache saved {money(c.cache_savings)} vs. uncached input")
    per_prompt = f"{money(report.cost_per_prompt)} per prompt, " if report.cost_per_prompt is not None else ""
    out.append(f"Unit cost:                  {per_prompt}{money(report.cost_per_turn)} per API call")
    if report.has_estimated_pricing:
        out.append("Warning:                    some turns use estimated rates (unrecognized model ID)")
    out.append("")

    # -- by model ------------------------------------------------------------
    if len(report.by_model) > 1 or s.subagents:
        out.append(heading("By model (including subagents)", WIDTH, "-"))
        out.append(table(
            ["Model", "Turns", "Context tokens", "Output tokens", "Cost", "Share"],
            [[m.model + (" *" if m.is_estimate else ""), m.turns, tokens(m.usage.context_tokens),
              tokens(m.usage.output_tokens), money(m.cost.total),
              pct(m.cost.total / report.cost_with_subagents.total if report.cost_with_subagents.total else 0)]
             for m in report.by_model],
        ))
        out.append("")

    # -- context -------------------------------------------------------------
    out.append(heading("Context window", WIDTH, "-"))
    out.append(f"Peak {tokens(report.peak_context)} tokens, mean {tokens(int(report.mean_context))} per call.")
    if len(report.phases) > 1:
        out.append("")
        out.append(table(
            ["Phase", "Turns", "Start ctx", "Peak ctx", "End ctx", "Cost"],
            [[p.index, f"{p.first_turn}-{p.last_turn}", tokens(p.start_context), tokens(p.peak_context),
              tokens(p.end_context), money(p.cost)] for p in report.phases],
        ))
        for comp in s.compactions:
            out.append(f"  compaction after turn {comp.after_turn} ({comp.trigger or 'unknown'}): "
                       f"{tokens(comp.pre_tokens)} -> {tokens(comp.post_tokens)} at {when(comp.timestamp)}")
    out.append("")

    # -- tools ---------------------------------------------------------------
    if report.tools:
        out.append(heading("Tool calls (this conversation)", WIDTH, "-"))
        out.append(table(
            ["Tool", "Calls", "Errors", "Result chars", "~Tokens", "Largest result"],
            [[t.name, t.calls, t.errors or "", f"{t.result_chars:,}", tokens(t.est_result_tokens),
              shorten(f"{t.largest_result_chars:,} {t.largest_target}", 34)] for t in report.tools[:12]],
        ))
        out.append("")

    # -- what if -------------------------------------------------------------
    if report.what_if:
        out.append(heading("Same conversation on another model (first-order estimate)", WIDTH, "-"))
        out.append(table(
            ["Model", "Cost", "vs. actual"],
            [[w.display, money(w.cost), f"{signed_money(w.delta)} ({w.delta_pct:+.0f}%)" if w.delta_pct is not None else signed_money(w.delta)]
             for w in report.what_if],
        ))
        out.append("")

    # -- findings ------------------------------------------------------------
    out.append(heading("Findings", WIDTH, "-"))
    savings = total_estimated_savings(findings)
    if savings >= 0.005:
        out.append(f"Estimated avoidable spend: ~{money(savings)} of {money(total)} ({pct(savings / total if total else 0)})")
        out.append("")
    out.append(render_findings(findings, findings_limit))
    out.append("")
    out.append(wrap(DISCLAIMER, WIDTH))
    return "\n".join(out)


def render_sessions_table(report: AggregateReport, limit: Optional[int] = None) -> str:
    rows = []
    sessions = report.sessions if limit is None else report.sessions[:limit]
    for s in sessions:
        rows.append([
            s.session_id[:8],
            when(s.ended_at, with_time=False),
            shorten(s.project_path.rstrip("/").split("/")[-1] if s.project_path else "-", 24),
            shorten(s.title or "-", 36),
            s.turns,
            tokens(s.peak_context),
            money(s.cost),
            money(s.subagent_cost) if s.subagent_cost else "",
        ])
    header = f"{report.session_count} sessions, {report.turn_count:,} API calls, {money(report.total_cost)} total"
    if report.total_subagent_cost:
        header += f" (subagents {money(report.total_subagent_cost)})"
    return header + "\n\n" + table(
        ["Session", "Last active", "Project", "Title", "Turns", "Peak ctx", "Cost", "Subagents"], rows,
        align=["l", "l", "l", "l", "r", "r", "r", "r"],
    )


def render_summary(report: AggregateReport, findings: Sequence[Finding],
                   findings_limit: Optional[int] = 10, days_label: str = "") -> str:
    out: List[str] = []
    out.append(heading(f"Claude Code usage summary{days_label}", WIDTH))
    out.append(f"Sessions: {report.session_count}   API calls: {report.turn_count:,}   "
               f"Total: {money(report.total_cost)}" +
               (f"   (subagents {money(report.total_subagent_cost)}, {pct(report.total_subagent_cost / report.total_cost)})"
                if report.total_cost and report.total_subagent_cost else ""))
    if report.daily:
        out.append(f"Active days: {len(report.daily)} ({report.daily[0].day} to {report.daily[-1].day}), "
                   f"average {money(report.total_cost / len(report.daily))} per active day")
    c = report.cost
    if c.total:
        out.append(f"Carrying context: {money(c.cache_read_cost)} ({pct(c.cache_read_cost / c.total)})   "
                   f"Cache writes: {money(c.cache_write_cost)}   Output: {money(c.output_cost)}   "
                   f"Fresh input: {money(c.input_cost)}")
    if report.has_estimated_pricing:
        out.append("Warning: some turns use estimated rates (unrecognized model ID)")
    out.append("")

    out.append(heading("By project", WIDTH, "-"))
    out.append(table(
        ["Project", "Sessions", "Turns", "Cost", "Subagents", "Last active"],
        [[shorten(p.project_path, 44), p.sessions, f"{p.turns:,}", money(p.cost),
          money(p.subagent_cost) if p.subagent_cost else "", when(p.last_active, with_time=False)]
         for p in report.by_project[:15]],
    ))
    out.append("")

    out.append(heading("By model", WIDTH, "-"))
    out.append(table(
        ["Model", "Turns", "Context tokens", "Output tokens", "Cost", "Share"],
        [[m.model + (" *" if m.is_estimate else ""), f"{m.turns:,}", tokens(m.usage.context_tokens),
          tokens(m.usage.output_tokens), money(m.cost.total), pct(m.cost.total / report.total_cost if report.total_cost else 0)]
         for m in report.by_model],
    ))
    out.append("")

    if report.what_if:
        out.append(heading("Everything on one model (first-order estimate)", WIDTH, "-"))
        out.append(table(
            ["Model", "Cost", "vs. actual"],
            [[w.display, money(w.cost), f"{signed_money(w.delta)} ({w.delta_pct:+.0f}%)" if w.delta_pct is not None else signed_money(w.delta)]
             for w in report.what_if],
        ))
        out.append("")

    if report.tools:
        out.append(heading("Tools by context consumed", WIDTH, "-"))
        out.append(table(
            ["Tool", "Calls", "Errors", "~Tokens into context"],
            [[t.name, f"{t.calls:,}", t.errors or "", tokens(t.est_result_tokens)] for t in report.tools[:10]],
        ))
        out.append("")

    out.append(heading("Top findings across sessions", WIDTH, "-"))
    out.append(render_findings(findings, findings_limit, show_provenance=True))
    out.append("")
    out.append(wrap(DISCLAIMER, WIDTH))
    return "\n".join(out)


def render_daily(report: AggregateReport, bar_width: int = 40) -> str:
    if not report.daily:
        return "No dated turns found."
    peak = max(d.cost for d in report.daily) or 1.0
    lines = [heading("Daily API-equivalent cost", WIDTH, "-")]
    for d in report.daily:
        bar = "#" * max(int(round(d.cost / peak * bar_width)), 1 if d.cost > 0 else 0)
        lines.append(f"{d.day}  {money(d.cost):>9}  {bar}")
    return "\n".join(lines)
