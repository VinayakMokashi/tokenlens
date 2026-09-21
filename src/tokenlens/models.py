"""Core data model shared by the parser, analysis, and presentation layers.

Everything here is a plain dataclass with no behaviour beyond simple
derived properties, so the objects serialize cleanly to JSON and are
easy to construct by hand in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, List, Optional


@dataclass
class TokenUsage:
    """Token counts for one API call, split the way Anthropic bills them.

    ``cache_write_5m_tokens`` and ``cache_write_1h_tokens`` come from the
    ``usage.cache_creation`` sub-object that Claude Code records. They are
    billed at different multipliers (1.25x and 2x the input rate), which
    is why they are kept apart instead of collapsed into one number.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def context_tokens(self) -> int:
        """Size of the prompt the model saw on this call.

        Every token in the prompt is billed exactly once as either fresh
        input, a cache read, or a cache write, so the sum of the three is
        the full context window occupancy at that moment. This is the
        number that grows over a session and is the main cost driver in
        Claude Code.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.context_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_5m_tokens=self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            cache_write_1h_tokens=self.cache_write_1h_tokens + other.cache_write_1h_tokens,
        )

    @classmethod
    def total(cls, items: Iterable["TokenUsage"]) -> "TokenUsage":
        acc = cls()
        for item in items:
            acc = acc + item
        return acc


@dataclass
class CostBreakdown:
    """Dollar cost of one usage record, by billing category."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    #: What ``cache_read_tokens`` would have cost as ordinary input. The
    #: difference between this and ``cache_read_cost`` is the money the
    #: cache saved on this call.
    uncached_read_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    @property
    def cache_savings(self) -> float:
        return self.uncached_read_cost - self.cache_read_cost

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
            cache_write_cost=self.cache_write_cost + other.cache_write_cost,
            uncached_read_cost=self.uncached_read_cost + other.uncached_read_cost,
        )

    @classmethod
    def total_of(cls, items: Iterable["CostBreakdown"]) -> "CostBreakdown":
        acc = cls()
        for item in items:
            acc = acc + item
        return acc


@dataclass
class ToolCall:
    """One ``tool_use`` block and, when found, its matching ``tool_result``."""

    id: str
    name: str
    #: Short human-readable description of the input (a file path, a
    #: command, a search pattern), used in findings and tables.
    target: str = ""
    input_chars: int = 0
    #: ``None`` when no tool_result was found (interrupted turn, or the
    #: result lives in a subagent transcript).
    result_chars: Optional[int] = None
    result_preview: str = ""
    is_error: bool = False


@dataclass
class Turn:
    """One billed API call, reconstructed from one or more JSONL lines.

    Claude Code writes each content block of an assistant response as its
    own line, all sharing the same ``message.id`` and repeating the same
    ``usage``. The parser folds them back into a single Turn.
    """

    index: int
    message_id: str
    timestamp: Optional[datetime]
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    speed: str = "standard"
    effort: str = ""
    stop_reason: str = ""
    request_id: str = ""
    thinking_chars: int = 0
    text_chars: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    #: True when the pricing table had no exact entry for ``model`` and
    #: fell back to a family default.
    pricing_is_estimate: bool = False

    @property
    def context_tokens(self) -> int:
        return self.usage.context_tokens

    @property
    def kind(self) -> str:
        """Coarse classification used for grouping in reports."""
        if self.tool_calls:
            return "tool_use"
        if self.thinking_chars and not self.text_chars:
            return "thinking"
        return "text"


@dataclass
class CompactionEvent:
    """A ``compact_boundary`` system line: Claude Code summarized history."""

    timestamp: Optional[datetime]
    trigger: str = ""
    pre_tokens: int = 0
    post_tokens: int = 0
    #: Index of the last turn before the compaction (0 if none).
    after_turn: int = 0

    @property
    def dropped_tokens(self) -> int:
        return max(self.pre_tokens - self.post_tokens, 0)


@dataclass
class ApiError:
    """An assistant line flagged ``isApiErrorMessage`` (rate limit, 5xx...)."""

    timestamp: Optional[datetime]
    status: Optional[int] = None
    error: str = ""
    message: str = ""
    #: Index of the last successful turn before the error.
    after_turn: int = 0


@dataclass
class Session:
    """A parsed transcript plus the subagent transcripts it spawned."""

    path: str
    session_id: str
    #: Real working directory, taken from the ``cwd`` field the transcript
    #: itself records. This is exact, unlike decoding the directory name.
    project_path: str = ""
    project_dir: str = ""
    title: str = ""
    first_prompt: str = ""
    git_branch: str = ""
    entrypoint: str = ""
    claude_versions: List[str] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)
    compactions: List[CompactionEvent] = field(default_factory=list)
    api_errors: List[ApiError] = field(default_factory=list)
    user_prompt_count: int = 0
    is_subagent: bool = False
    agent_id: str = ""
    #: Set for subagents spawned by the Workflow tool, whose transcripts sit
    #: under ``subagents/workflows/wf_<id>/``.
    workflow_id: str = ""
    parent_session_id: str = ""
    subagents: List["Session"] = field(default_factory=list)
    #: Set by the parser when lines could not be decoded as JSON.
    skipped_lines: int = 0

    # -- derived -------------------------------------------------------

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage.total(t.usage for t in self.turns)

    @property
    def cost(self) -> CostBreakdown:
        return CostBreakdown.total_of(t.cost for t in self.turns)

    @property
    def total_cost(self) -> float:
        return self.cost.total

    @property
    def subagent_cost(self) -> float:
        return sum(s.total_cost_with_subagents for s in self.subagents)

    @property
    def total_cost_with_subagents(self) -> float:
        return self.total_cost + self.subagent_cost

    @property
    def models(self) -> List[str]:
        seen: List[str] = []
        for t in self.turns:
            if t.model and t.model not in seen:
                seen.append(t.model)
        return seen

    @property
    def started_at(self) -> Optional[datetime]:
        stamps = [t.timestamp for t in self.turns if t.timestamp]
        return min(stamps) if stamps else None

    @property
    def ended_at(self) -> Optional[datetime]:
        stamps = [t.timestamp for t in self.turns if t.timestamp]
        return max(stamps) if stamps else None

    @property
    def wall_duration(self) -> Optional[timedelta]:
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None

    def active_duration(self, idle_gap: timedelta = timedelta(minutes=30)) -> timedelta:
        """Time spent actually working, ignoring gaps longer than ``idle_gap``.

        Sessions are frequently resumed days or weeks later, so wall-clock
        duration is meaningless for judging throughput.
        """
        stamps = sorted(t.timestamp for t in self.turns if t.timestamp)
        active = timedelta()
        for earlier, later in zip(stamps, stamps[1:]):
            gap = later - earlier
            if gap <= idle_gap:
                active += gap
        return active

    def all_turns(self) -> List[Turn]:
        """Turns from this session and every subagent, in one flat list."""
        result = list(self.turns)
        for sub in self.subagents:
            result.extend(sub.all_turns())
        return result
