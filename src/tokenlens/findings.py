"""Rule-based detection of avoidable spend in a session.

Each rule inspects a parsed :class:`~tokenlens.models.Session` and emits
:class:`Finding` records with a severity, the turns involved, a plain
explanation, and where possible an estimate of the dollars that could
have been saved. Estimates are deliberately conservative and always
explained in the finding text. Findings that only *explain* a cost
(cache expiry after a break, a model switch, API errors) carry no
saving so the headline "avoidable" figure stays honest.

The estimates use one idea repeatedly: a token added to the context is
paid for once as a cache write and then again on **every following turn
until the next compaction** as a cache read. That "carry" multiplier is
what makes a 30 KB file read in turn 5 of a 400-turn session expensive,
and it is what most token analyzers miss when they price a tool result
as if it were seen once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .analysis import CHARS_PER_TOKEN, phases as split_phases
from .formatting import money, tokens as fmt_tokens
from .models import Session, Turn
from .pricing import DEFAULT_TABLE, MILLION, PricingTable, Rates

SEVERITY_ORDER = {"critical": 3, "warning": 2, "info": 1}

#: Rules whose saving is the carry cost of specific tokens. Their savings
#: overlap with a context_bloat finding covering the same turns.
CARRY_RULES = ("large_tool_result", "duplicate_read")


@dataclass
class Thresholds:
    """Tunable knobs for every rule, with defaults chosen against real
    sessions so that the output is short and each item is worth reading."""

    large_result_chars: int = 16_000       # ~4K tokens
    huge_result_chars: int = 100_000       # ~25K tokens
    context_bloat_tokens: int = 400_000
    healthy_context_tokens: int = 60_000   # what a compaction typically leaves behind
    error_streak: int = 3
    cache_miss_min_write_tokens: int = 20_000
    thinking_share_warn: float = 0.60
    subagent_share_info: float = 0.30
    duplicate_read_min_chars: int = 2_000


@dataclass
class Finding:
    rule: str
    severity: str
    title: str
    detail: str
    turns: List[int] = field(default_factory=list)
    est_tokens_saved: int = 0
    est_dollars_saved: float = 0.0
    evidence: Dict[str, object] = field(default_factory=dict)
    session_id: str = ""
    project_path: str = ""

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


# -- helpers -----------------------------------------------------------------


def _phase_end_index(turn: Turn, phase_bounds: Sequence[Tuple[int, int]]) -> int:
    for first, last in phase_bounds:
        if first <= turn.index <= last:
            return last
    return turn.index


def _carry_cost(token_count: int, rates: Rates, carry_turns: int) -> float:
    """Cost to write ``token_count`` tokens once (1h tier, Claude Code's
    default) and re-read them on ``carry_turns`` later turns."""
    write = token_count / MILLION * rates.cache_write_1h
    reads = token_count / MILLION * rates.cache_read * carry_turns
    return write + reads


def _read_path(target: str) -> str:
    """The file path part of a Read target, without the slice annotation."""
    return target.split(" [offset=")[0].strip()


# -- rules -------------------------------------------------------------------


def rule_large_tool_results(session: Session, pricing: PricingTable, th: Thresholds,
                            phase_bounds: Sequence[Tuple[int, int]]) -> List[Finding]:
    out: List[Finding] = []
    for turn in session.turns:
        rates = pricing.resolve(turn.model, turn.speed).rates
        phase_end = _phase_end_index(turn, phase_bounds)
        carry_turns = phase_end - turn.index
        for call in turn.tool_calls:
            chars = call.result_chars or 0
            if chars < th.large_result_chars:
                continue
            token_count = chars // CHARS_PER_TOKEN
            dollars = _carry_cost(token_count, rates, carry_turns)
            severity = "warning" if chars >= th.huge_result_chars else "info"
            hint = {
                "Read": "Read a slice with offset/limit, or Grep for the lines you need.",
                "Bash": "Pipe through head/tail/grep, or redirect to a file and read a slice.",
                "PowerShell": "Use Select-Object -First/-Last, or redirect to a file and read a slice.",
                "Grep": "Narrow the pattern or path, or use files_with_matches mode.",
                "WebFetch": "Ask for a targeted extraction in the prompt instead of the full page.",
            }.get(call.name, "Ask for a summary or a narrower result.")
            out.append(Finding(
                rule="large_tool_result",
                severity=severity,
                title=f"{call.name} pulled ~{fmt_tokens(token_count)} tokens into context",
                detail=(
                    f"Turn {turn.index}: {call.name}({call.target}) returned {chars:,} characters. "
                    f"Those tokens were written to cache once and then re-read on each of the "
                    f"{carry_turns} turns that followed before the next compaction, for about "
                    f"{money(dollars)} in total. {hint}"
                ),
                turns=[turn.index],
                est_tokens_saved=token_count,
                est_dollars_saved=dollars,
                evidence={"tool": call.name, "target": call.target, "chars": chars,
                          "carry_turns": carry_turns, "carry_until": phase_end},
            ))
    return out


def rule_duplicate_reads(session: Session, pricing: PricingTable, th: Thresholds,
                         phase_bounds: Sequence[Tuple[int, int]]) -> List[Finding]:
    """A Read with the *same target* (path and slice) as an earlier Read,
    with no Edit/Write to that file in between, re-sends content that is
    still in the context. Two different slices of one file are not
    duplicates; the large-result rule actively recommends slicing."""
    out: List[Finding] = []
    last_read_turn: Dict[str, int] = {}
    for turn in session.turns:
        rates = pricing.resolve(turn.model, turn.speed).rates
        phase_end = _phase_end_index(turn, phase_bounds)
        carry_turns = phase_end - turn.index
        for call in turn.tool_calls:
            if call.name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
                edited = _read_path(call.target)
                for key in [k for k in last_read_turn if _read_path(k) == edited]:
                    del last_read_turn[key]
                continue
            if call.name != "Read" or not call.target:
                continue
            chars = call.result_chars or 0
            previous = last_read_turn.get(call.target)
            if previous is not None and chars >= th.duplicate_read_min_chars:
                token_count = chars // CHARS_PER_TOKEN
                dollars = _carry_cost(token_count, rates, carry_turns)
                out.append(Finding(
                    rule="duplicate_read",
                    severity="warning",
                    title=f"Re-read an unchanged file ({fmt_tokens(token_count)} tokens)",
                    detail=(
                        f"Turn {turn.index} read {call.target} again although it was read in turn {previous} "
                        f"and not modified since. The earlier copy was still in context, so this added "
                        f"{chars:,} duplicate characters worth about {money(dollars)} over the rest "
                        f"of the phase. Reference the earlier read or ask for a specific line range."
                    ),
                    turns=[previous, turn.index],
                    est_tokens_saved=token_count,
                    est_dollars_saved=dollars,
                    evidence={"path": call.target, "first_turn": previous, "chars": chars,
                              "carry_turns": carry_turns, "carry_until": phase_end},
                ))
            last_read_turn[call.target] = turn.index
    return out


def rule_context_bloat(session: Session, pricing: PricingTable, th: Thresholds) -> List[Finding]:
    """Runs of turns carrying a very large context. The saving is what the
    cache reads above a healthy context size cost."""
    out: List[Finding] = []
    run: List[Turn] = []

    def flush() -> None:
        if not run:
            return
        excess_cost = 0.0
        excess_tokens = 0
        for t in run:
            rates = pricing.resolve(t.model, t.speed).rates
            excess = max(t.usage.cache_read_tokens - th.healthy_context_tokens, 0)
            excess_tokens += excess
            excess_cost += excess / MILLION * rates.cache_read
        peak = max(t.context_tokens for t in run)
        carry = sum(t.cost.cache_read_cost for t in run)
        severity = "critical" if excess_cost >= 10 else "warning"
        out.append(Finding(
            rule="context_bloat",
            severity=severity,
            title=f"{len(run)} turns ran with context above {fmt_tokens(th.context_bloat_tokens)} tokens",
            detail=(
                f"Turns {run[0].index}-{run[-1].index} carried between "
                f"{fmt_tokens(min(t.context_tokens for t in run))} and {fmt_tokens(peak)} tokens of context. "
                f"Re-reading that context from cache cost {money(carry)} across the run. "
                f"Had the context been compacted to ~{fmt_tokens(th.healthy_context_tokens)} tokens at turn "
                f"{run[0].index} (with /compact, or by starting a fresh session for the next task), "
                f"roughly {money(excess_cost)} of that would have been avoided."
            ),
            turns=[run[0].index, run[-1].index],
            est_tokens_saved=excess_tokens,
            est_dollars_saved=excess_cost,
            evidence={"peak_context": peak, "turns": len(run), "carry_cost": carry},
        ))

    for turn in session.turns:
        # Once a run has started, it continues until context falls clearly
        # below the threshold (10% margin), so a single turn dipping a few
        # hundred tokens under the line does not split one long run in two.
        limit = th.context_bloat_tokens * (0.9 if run else 1.0)
        if turn.context_tokens >= limit:
            run.append(turn)
        else:
            flush()
            run = []
    flush()
    return out


def rule_context_rewrites(session: Session, pricing: PricingTable, th: Thresholds) -> List[Finding]:
    """A turn that reads far less from cache than the previous turn's
    context, and writes a large amount instead, re-wrote the conversation
    at the 1-hour rate rather than reading it at a tenth of the price.

    The test is relative to the previous context rather than "zero cache
    reads" because Claude Code keeps a shared prefix (system prompt and
    tools, ~28K tokens) cached across conversations, so a miss on the
    conversation itself still shows a small read.

    Causes are reported separately: a model switch (caches are per
    model), or a cache miss after a break or a change to the prompt
    prefix. Neither is counted as avoidable; the finding explains the
    cost so the user can weigh it."""
    out: List[Finding] = []
    previous: Optional[Turn] = None
    for turn in session.turns:
        u = turn.usage
        if (previous is not None
                and u.cache_write_tokens >= th.cache_miss_min_write_tokens
                and u.cache_read_tokens < previous.context_tokens * 0.5):
            rates = pricing.resolve(turn.model, turn.speed).rates
            as_read = u.cache_write_tokens / MILLION * rates.cache_read
            paid = turn.cost.cache_write_cost
            extra = max(paid - as_read, 0.0)
            evidence = {"rewritten_tokens": u.cache_write_tokens, "paid": paid,
                        "as_cache_read": as_read, "extra_cost": extra}

            if previous.model and turn.model and previous.model != turn.model:
                out.append(Finding(
                    rule="model_switch",
                    severity="info",
                    title=f"Model switch re-wrote {fmt_tokens(u.cache_write_tokens)} tokens of context",
                    detail=(
                        f"Turn {turn.index} moved from {previous.model} to {turn.model}. Prompt caches are "
                        f"per model, so the whole {u.cache_write_tokens:,}-token context was written again "
                        f"for {money(paid)} instead of {money(as_read)} as a cache read, about {money(extra)} "
                        f"extra. Switching models is often worth it; just do it at a natural break rather "
                        f"than back and forth."
                    ),
                    turns=[turn.index],
                    evidence={**evidence, "from_model": previous.model, "to_model": turn.model},
                ))
            else:
                cause = ("The conversation's prompt cache was not reused, so the context was "
                         "written again")
                if previous.timestamp and turn.timestamp:
                    delta: timedelta = turn.timestamp - previous.timestamp
                    minutes = delta.total_seconds() / 60
                    evidence["gap_minutes"] = round(minutes, 1)
                    if minutes >= 60:
                        cause = (f"It came {minutes / 60:.1f} hours after the previous call, longer than "
                                 f"the one-hour cache lifetime, so the context was written again")
                    else:
                        cause = (f"It came only {minutes:.0f} minutes after the previous call, so the cache "
                                 f"was most likely invalidated rather than expired: something early in the "
                                 f"prompt changed (tool list, MCP servers, settings, effort level) or the "
                                 f"entry was evicted. The context was written again")
                out.append(Finding(
                    rule="cache_miss",
                    severity="info",
                    title=(f"Cache miss: {fmt_tokens(u.cache_write_tokens)} tokens re-written "
                           f"(~{money(extra)} extra)"),
                    detail=(
                        f"Turn {turn.index} read only {u.cache_read_tokens:,} tokens from cache although the "
                        f"previous call carried {previous.context_tokens:,}. {cause} for {money(paid)} "
                        f"instead of {money(as_read)} as a cache read. After a long break this is "
                        f"unavoidable; if misses repeat within minutes, look for settings or tools that "
                        f"change mid-session."
                    ),
                    turns=[turn.index],
                    evidence=evidence,
                ))
        previous = turn
    return out


def rule_error_streaks(session: Session, th: Thresholds) -> List[Finding]:
    out: List[Finding] = []
    streak: List[Tuple[int, str, str]] = []  # (turn, tool, target)

    def flush() -> None:
        if len(streak) >= th.error_streak:
            tools = sorted({s[1] for s in streak})
            out.append(Finding(
                rule="error_streak",
                severity="warning",
                title=f"{len(streak)} consecutive failing tool calls",
                detail=(
                    f"Turns {streak[0][0]}-{streak[-1][0]}: {', '.join(tools)} failed {len(streak)} times in a row "
                    f"(last target: {streak[-1][2] or 'n/a'}). Each retry re-sends the full context. When a "
                    f"command keeps failing, stop and fix the environment or give the model the missing detail "
                    f"instead of letting it retry."
                ),
                turns=sorted({s[0] for s in streak}),
                evidence={"tools": tools, "count": len(streak)},
            ))

    for turn in session.turns:
        for call in turn.tool_calls:
            if call.is_error:
                streak.append((turn.index, call.name, call.target))
            else:
                flush()
                streak = []
    flush()
    return out


def rule_api_errors(session: Session) -> List[Finding]:
    if not session.api_errors:
        return []
    by_status: Dict[str, int] = {}
    for err in session.api_errors:
        key = f"{err.status or 'unknown'} {err.error}".strip()
        by_status[key] = by_status.get(key, 0) + 1
    summary = ", ".join(f"{count}x {key}" for key, count in sorted(by_status.items()))
    return [Finding(
        rule="api_errors",
        severity="info",
        title=f"{len(session.api_errors)} API error{'s' if len(session.api_errors) != 1 else ''} during the session",
        detail=(
            f"Claude Code logged {summary}. These are not billed, but rate-limit errors mean work "
            f"stalled until the limit reset. If they cluster, spread heavy sessions out or reduce "
            f"context size so each turn consumes less of the quota."
        ),
        turns=[e.after_turn for e in session.api_errors if e.after_turn],
        evidence={"by_status": by_status},
    )]


def rule_thinking_share(session: Session, th: Thresholds) -> List[Finding]:
    output = session.usage.output_tokens
    if output < 5_000:
        return []
    # Thinking is billed as output; the character-based estimate can
    # overshoot on dense text, so never claim more than 100%.
    thinking_tokens = min(sum(t.thinking_chars for t in session.turns) // CHARS_PER_TOKEN, output)
    share = thinking_tokens / output if output else 0.0
    if share < th.thinking_share_warn:
        return []
    efforts: Dict[str, int] = {}
    for t in session.turns:
        if t.effort:
            efforts[t.effort] = efforts.get(t.effort, 0) + 1
    effort_text = ", ".join(f"{k}: {v} turns" for k, v in sorted(efforts.items(), key=lambda kv: -kv[1])) or "not recorded"
    thinking_cost = sum(t.cost.output_cost for t in session.turns) * share
    return [Finding(
        rule="thinking_share",
        severity="info",
        title=f"About {share:.0%} of output tokens were thinking",
        detail=(
            f"Roughly {fmt_tokens(thinking_tokens)} of {fmt_tokens(output)} output tokens were extended "
            f"thinking (about {money(thinking_cost)}). Effort levels used: {effort_text}. Thinking is "
            f"valuable on hard problems; for routine edits and mechanical steps a lower effort setting "
            f"(/effort or the model picker) cuts this without hurting results."
        ),
        evidence={"thinking_tokens": thinking_tokens, "output_tokens": output, "efforts": efforts},
    )]


def rule_subagent_share(session: Session, th: Thresholds) -> List[Finding]:
    total = session.total_cost_with_subagents
    if not session.subagents or total <= 0:
        return []
    share = session.subagent_cost / total
    if share < th.subagent_share_info:
        return []
    return [Finding(
        rule="subagent_share",
        severity="info",
        title=f"Subagents accounted for {share:.0%} of this session's cost",
        detail=(
            f"{len(session.subagents)} subagent transcripts cost {money(session.subagent_cost)} of the "
            f"{money(total)} total. Subagents start with a fresh context, which is cheap per turn, "
            f"but each one re-reads the files it needs. Check that parallel agents are not all reading the "
            f"same large files, and prefer a cheaper model for mechanical sub-tasks."
        ),
        evidence={"subagents": len(session.subagents), "subagent_cost": session.subagent_cost},
    )]


def rule_estimated_pricing(session: Session) -> List[Finding]:
    models = sorted({t.model for t in session.all_turns() if t.pricing_is_estimate and t.model})
    if not models:
        return []
    return [Finding(
        rule="estimated_pricing",
        severity="warning",
        title="Some costs are estimates: unrecognized model ID",
        detail=(
            f"The pricing table has no exact entry for: {', '.join(models)}. Their turns were priced at "
            f"the closest family rate. Pass a JSON overrides file with --pricing to set exact rates."
        ),
        evidence={"models": models},
    )]


# -- entry point ---------------------------------------------------------------


def detect_findings(session: Session, pricing: PricingTable = DEFAULT_TABLE,
                    thresholds: Optional[Thresholds] = None) -> List[Finding]:
    th = thresholds or Thresholds()
    bounds = [(p.first_turn, p.last_turn) for p in split_phases(session)]

    findings: List[Finding] = []
    findings += rule_estimated_pricing(session)
    findings += rule_context_bloat(session, pricing, th)
    findings += rule_duplicate_reads(session, pricing, th, bounds)
    findings += rule_large_tool_results(session, pricing, th, bounds)
    findings += rule_context_rewrites(session, pricing, th)
    findings += rule_error_streaks(session, th)
    findings += rule_thinking_share(session, th)
    findings += rule_subagent_share(session, th)
    findings += rule_api_errors(session)

    for f in findings:
        f.session_id = session.session_id
        f.project_path = session.project_path

    findings.sort(key=lambda f: (f.severity_rank, f.est_dollars_saved), reverse=True)
    return findings


def total_estimated_savings(findings: Sequence[Finding]) -> float:
    """Sum of savings without double counting.

    A context_bloat finding already prices every excess token carried
    across its turn range, so a large_tool_result or duplicate_read whose
    carry window overlaps that range *in the same session* is not added
    on top. Findings from other sessions, or from turns outside the bloat
    range, count in full. Works on a single session's findings and on a
    pooled cross-session list alike.
    """
    bloat_ranges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for f in findings:
        if f.rule == "context_bloat" and len(f.turns) >= 2:
            bloat_ranges[f.session_id].append((f.turns[0], f.turns[-1]))

    total = 0.0
    for f in findings:
        if f.rule in CARRY_RULES and f.turns:
            origin = f.turns[-1]
            carry_until = int(f.evidence.get("carry_until", origin))  # type: ignore[arg-type]
            if any(origin <= last and carry_until >= first
                   for first, last in bloat_ranges.get(f.session_id, ())):
                continue
        total += f.est_dollars_saved
    return total
