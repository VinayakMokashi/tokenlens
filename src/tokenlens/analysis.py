"""Derive explanatory metrics from parsed sessions.

The parser answers "what happened"; this module answers "why did it cost
that". The central idea is the **context series**: for every turn, the
size of the prompt the model received. In Claude Code that prompt is the
whole conversation so far, re-sent on every call, and almost all of it
is served from cache. The cache-read charge for re-sending it is the
*carry cost* of the context, and separating carry cost from the cost of
new work (fresh input, cache writes, output) is what makes long sessions
understandable.

Everything here is pure computation over dataclasses so it can be unit
tested without files.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import CostBreakdown, Session, TokenUsage, Turn
from .pricing import DEFAULT_TABLE, PricingTable

CHARS_PER_TOKEN = 4  # rough English/code average, used only for tool-result estimates

WHAT_IF_MODELS: Tuple[str, ...] = (
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
)


@dataclass
class ModelUsage:
    model: str
    display: str
    turns: int
    usage: TokenUsage
    cost: CostBreakdown
    is_estimate: bool = False


@dataclass
class ToolStats:
    name: str
    calls: int = 0
    errors: int = 0
    result_chars: int = 0
    largest_result_chars: int = 0
    largest_target: str = ""

    @property
    def est_result_tokens(self) -> int:
        return self.result_chars // CHARS_PER_TOKEN


@dataclass
class ContextPoint:
    turn_index: int
    timestamp: Optional[datetime]
    model: str
    context_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    input_tokens: int
    output_tokens: int
    cost: float
    carry_cost: float
    compaction_before: bool = False


@dataclass
class Phase:
    """Turns between two compactions (or session start/end)."""

    index: int
    first_turn: int
    last_turn: int
    turns: int
    cost: float
    peak_context: int
    start_context: int
    end_context: int


@dataclass
class WhatIf:
    model: str
    display: str
    cost: float
    delta: float

    @property
    def delta_pct(self) -> Optional[float]:
        base = self.cost - self.delta
        return (self.delta / base * 100.0) if base else None


@dataclass
class DailyCost:
    day: date
    cost: float
    turns: int
    output_tokens: int


@dataclass
class SessionReport:
    session: Session
    usage: TokenUsage
    cost: CostBreakdown
    usage_with_subagents: TokenUsage
    cost_with_subagents: CostBreakdown
    by_model: List[ModelUsage]
    by_kind: Dict[str, Tuple[int, float]]
    tools: List[ToolStats]
    context: List[ContextPoint]
    phases: List[Phase]
    what_if: List[WhatIf]
    daily: List[DailyCost]
    has_estimated_pricing: bool

    # -- convenience metrics -------------------------------------------

    @property
    def turn_count(self) -> int:
        return len(self.session.turns)

    @property
    def peak_context(self) -> int:
        return max((p.context_tokens for p in self.context), default=0)

    @property
    def mean_context(self) -> float:
        return (sum(p.context_tokens for p in self.context) / len(self.context)) if self.context else 0.0

    @property
    def carry_cost(self) -> float:
        """Money spent re-reading existing context from cache."""
        return self.cost.cache_read_cost

    @property
    def new_work_cost(self) -> float:
        """Money spent on fresh input, cache writes, and output."""
        return self.cost.input_cost + self.cost.cache_write_cost + self.cost.output_cost

    @property
    def carry_share(self) -> float:
        total = self.cost.total
        return (self.carry_cost / total) if total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache."""
        prompt = self.usage.context_tokens
        return (self.usage.cache_read_tokens / prompt) if prompt else 0.0

    @property
    def cost_per_prompt(self) -> Optional[float]:
        n = self.session.user_prompt_count
        return (self.cost.total / n) if n else None

    @property
    def cost_per_turn(self) -> float:
        return (self.cost.total / self.turn_count) if self.turn_count else 0.0


# -- building blocks ------------------------------------------------------


def usage_by_model(turns: Iterable[Turn], pricing: PricingTable = DEFAULT_TABLE) -> List[ModelUsage]:
    buckets: "OrderedDict[str, ModelUsage]" = OrderedDict()
    for turn in turns:
        model = turn.model or "(unknown)"
        entry = buckets.get(model)
        if entry is None:
            res = pricing.resolve(turn.model, turn.speed)
            entry = ModelUsage(model=model, display=res.spec.display, turns=0,
                               usage=TokenUsage(), cost=CostBreakdown(), is_estimate=res.is_estimate)
            buckets[model] = entry
        entry.turns += 1
        entry.usage = entry.usage + turn.usage
        entry.cost = entry.cost + turn.cost
    return sorted(buckets.values(), key=lambda m: m.cost.total, reverse=True)


def usage_by_kind(turns: Iterable[Turn]) -> Dict[str, Tuple[int, float]]:
    counts: Dict[str, int] = defaultdict(int)
    costs: Dict[str, float] = defaultdict(float)
    for turn in turns:
        counts[turn.kind] += 1
        costs[turn.kind] += turn.cost.total
    return {k: (counts[k], costs[k]) for k in sorted(counts, key=lambda k: costs[k], reverse=True)}


def tool_stats(turns: Iterable[Turn]) -> List[ToolStats]:
    stats: Dict[str, ToolStats] = {}
    for turn in turns:
        for call in turn.tool_calls:
            entry = stats.setdefault(call.name, ToolStats(name=call.name))
            entry.calls += 1
            if call.is_error:
                entry.errors += 1
            chars = call.result_chars or 0
            entry.result_chars += chars
            if chars > entry.largest_result_chars:
                entry.largest_result_chars = chars
                entry.largest_target = call.target
    return sorted(stats.values(), key=lambda s: s.result_chars, reverse=True)


def context_series(session: Session) -> List[ContextPoint]:
    compaction_after = {c.after_turn for c in session.compactions}
    points: List[ContextPoint] = []
    for turn in session.turns:
        points.append(ContextPoint(
            turn_index=turn.index,
            timestamp=turn.timestamp,
            model=turn.model,
            context_tokens=turn.context_tokens,
            cache_read_tokens=turn.usage.cache_read_tokens,
            cache_write_tokens=turn.usage.cache_write_tokens,
            input_tokens=turn.usage.input_tokens,
            output_tokens=turn.usage.output_tokens,
            cost=turn.cost.total,
            carry_cost=turn.cost.cache_read_cost,
            compaction_before=(turn.index - 1) in compaction_after,
        ))
    return points


def phases(session: Session) -> List[Phase]:
    """Split the turn list at compaction boundaries."""
    if not session.turns:
        return []
    boundaries = sorted({c.after_turn for c in session.compactions if 0 < c.after_turn < len(session.turns)})
    starts = [0] + boundaries
    ends = boundaries + [len(session.turns)]
    result: List[Phase] = []
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        chunk = session.turns[start:end]
        if not chunk:
            continue
        result.append(Phase(
            index=i,
            first_turn=chunk[0].index,
            last_turn=chunk[-1].index,
            turns=len(chunk),
            cost=sum(t.cost.total for t in chunk),
            peak_context=max(t.context_tokens for t in chunk),
            start_context=chunk[0].context_tokens,
            end_context=chunk[-1].context_tokens,
        ))
    return result


def what_if_costs(turns: Sequence[Turn], pricing: PricingTable = DEFAULT_TABLE,
                  candidates: Sequence[str] = WHAT_IF_MODELS) -> List[WhatIf]:
    """Re-price every turn at another model's rates.

    Token counts are kept as recorded, so this is the cost of the *same
    conversation shape* on a different rate card. Different models
    tokenize slightly differently and may take more or fewer turns to
    finish, so treat the result as a first-order estimate.
    """
    actual = sum(t.cost.total for t in turns)
    results: List[WhatIf] = []
    for key in candidates:
        spec = pricing.spec_for_key(key)
        if spec is None:
            continue
        total = sum(pricing.price_with_rates(t.usage, spec.rates).total for t in turns)
        results.append(WhatIf(model=key, display=spec.display, cost=total, delta=total - actual))
    return results


def daily_costs(turns: Iterable[Turn]) -> List[DailyCost]:
    days: Dict[date, DailyCost] = {}
    for turn in turns:
        if turn.timestamp is None:
            continue
        day = turn.timestamp.date()
        entry = days.setdefault(day, DailyCost(day=day, cost=0.0, turns=0, output_tokens=0))
        entry.cost += turn.cost.total
        entry.turns += 1
        entry.output_tokens += turn.usage.output_tokens
    return [days[d] for d in sorted(days)]


def analyze_session(session: Session, pricing: PricingTable = DEFAULT_TABLE) -> SessionReport:
    all_turns = session.all_turns()
    return SessionReport(
        session=session,
        usage=session.usage,
        cost=session.cost,
        usage_with_subagents=TokenUsage.total(t.usage for t in all_turns),
        cost_with_subagents=CostBreakdown.total_of(t.cost for t in all_turns),
        by_model=usage_by_model(all_turns, pricing),
        by_kind=usage_by_kind(session.turns),
        tools=tool_stats(session.turns),
        context=context_series(session),
        phases=phases(session),
        what_if=what_if_costs(session.turns, pricing),
        daily=daily_costs(session.turns),
        has_estimated_pricing=any(t.pricing_is_estimate for t in all_turns),
    )


# -- cross-session aggregation --------------------------------------------


@dataclass
class ProjectCost:
    project_path: str
    sessions: int
    turns: int
    cost: float
    subagent_cost: float
    last_active: Optional[datetime]


@dataclass
class SessionSummary:
    session_id: str
    project_path: str
    title: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    turns: int
    prompts: int
    cost: float
    subagent_cost: float
    subagents: int
    peak_context: int
    compactions: int
    api_errors: int
    models: List[str]
    is_subagent: bool = False


@dataclass
class AggregateReport:
    sessions: List[SessionSummary]
    total_cost: float
    total_subagent_cost: float
    usage: TokenUsage
    cost: CostBreakdown
    by_project: List[ProjectCost]
    by_model: List[ModelUsage]
    daily: List[DailyCost]
    tools: List[ToolStats]
    what_if: List[WhatIf]
    has_estimated_pricing: bool

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def turn_count(self) -> int:
        return sum(s.turns for s in self.sessions)


def summarize_session(session: Session) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        project_path=session.project_path,
        title=session.title or session.first_prompt,
        started_at=session.started_at,
        ended_at=session.ended_at,
        turns=len(session.turns),
        prompts=session.user_prompt_count,
        cost=session.total_cost,
        subagent_cost=session.subagent_cost,
        subagents=len(session.subagents),
        peak_context=max((t.context_tokens for t in session.turns), default=0),
        compactions=len(session.compactions),
        api_errors=len(session.api_errors),
        models=session.models,
        is_subagent=session.is_subagent,
    )


def aggregate(sessions: Sequence[Session], pricing: PricingTable = DEFAULT_TABLE) -> AggregateReport:
    all_turns: List[Turn] = []
    main_turns: List[Turn] = []
    projects: Dict[str, ProjectCost] = {}
    summaries: List[SessionSummary] = []

    for session in sessions:
        turns = session.all_turns()
        all_turns.extend(turns)
        main_turns.extend(session.turns)
        summaries.append(summarize_session(session))
        key = session.project_path or session.project_dir or "(unknown)"
        entry = projects.get(key)
        if entry is None:
            entry = ProjectCost(project_path=key, sessions=0, turns=0, cost=0.0,
                                subagent_cost=0.0, last_active=None)
            projects[key] = entry
        entry.sessions += 1
        entry.turns += len(session.turns)
        entry.cost += session.total_cost_with_subagents
        entry.subagent_cost += session.subagent_cost
        ended = session.ended_at
        if ended and (entry.last_active is None or ended > entry.last_active):
            entry.last_active = ended

    summaries.sort(key=lambda s: (s.ended_at or datetime.min.replace(tzinfo=None)).timestamp()
                   if s.ended_at else 0, reverse=True)

    return AggregateReport(
        sessions=summaries,
        total_cost=sum(t.cost.total for t in all_turns),
        total_subagent_cost=sum(s.subagent_cost for s in sessions),
        usage=TokenUsage.total(t.usage for t in all_turns),
        cost=CostBreakdown.total_of(t.cost for t in all_turns),
        by_project=sorted(projects.values(), key=lambda p: p.cost, reverse=True),
        by_model=usage_by_model(all_turns, pricing),
        daily=daily_costs(all_turns),
        tools=tool_stats(main_turns),
        what_if=what_if_costs(all_turns, pricing),
        has_estimated_pricing=any(t.pricing_is_estimate for t in all_turns),
    )


def rolling_window(daily: Sequence[DailyCost], days: int) -> List[DailyCost]:
    """Restrict a daily series to the last ``days`` calendar days."""
    if not daily:
        return []
    cutoff = daily[-1].day - timedelta(days=days - 1)
    return [d for d in daily if d.day >= cutoff]
