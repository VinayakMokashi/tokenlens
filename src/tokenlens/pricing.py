"""Model-aware pricing for Claude API usage.

Rates are US dollars per million tokens and follow Anthropic's published
first-party API pricing. Cache reads and writes are expressed as
multipliers of the input rate because that relationship is stable across
models, while the base rates change with each generation:

- cache read: 0.10x input (0.025x on Claude Fable 5.1)
- cache write, 5-minute TTL: 1.25x input
- cache write, 1-hour TTL: 2.00x input

Claude Code uses the 1-hour TTL for most of its cache writes, which is why
the two tiers are priced separately instead of assuming 1.25x.

Model IDs are resolved by pattern so that dated snapshots
(``claude-opus-4-5-20251101``), provider prefixes (``anthropic.``,
``us.anthropic.``), Vertex-style ``@`` versions, and routing suffixes such
as ``[1m]`` all map to the right entry. Unknown IDs fall back to their
family (fable/opus/sonnet/haiku) and are flagged as estimates rather than
silently priced at some arbitrary tier.

The table can be extended or overridden at runtime with a JSON file (see
:meth:`PricingTable.with_overrides`) so users are not stuck when a new
model ships before this package is updated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import CostBreakdown, TokenUsage

MILLION = 1_000_000

SYNTHETIC_MODEL = "<synthetic>"


@dataclass(frozen=True)
class Rates:
    """Dollar rates per million tokens for one model and speed."""

    input: float
    output: float
    cache_read_multiplier: float = 0.10
    cache_write_5m_multiplier: float = 1.25
    cache_write_1h_multiplier: float = 2.00

    @property
    def cache_read(self) -> float:
        return self.input * self.cache_read_multiplier

    @property
    def cache_write_5m(self) -> float:
        return self.input * self.cache_write_5m_multiplier

    @property
    def cache_write_1h(self) -> float:
        return self.input * self.cache_write_1h_multiplier

    def to_dict(self) -> Dict[str, float]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write_5m": self.cache_write_5m,
            "cache_write_1h": self.cache_write_1h,
        }


ZERO_RATES = Rates(input=0.0, output=0.0, cache_read_multiplier=0.0,
                   cache_write_5m_multiplier=0.0, cache_write_1h_multiplier=0.0)


@dataclass(frozen=True)
class ModelSpec:
    """One priced model. ``pattern`` is a regex tested against the
    normalized model ID; the first matching spec wins, so specific
    versions must be listed before their family fallback."""

    key: str
    display: str
    family: str
    pattern: str
    rates: Rates
    #: Rates when ``usage.speed == "fast"``; ``None`` means fast mode is
    #: not offered (or its price is unknown) and standard rates apply.
    fast_rates: Optional[Rates] = None


# Ordered: most specific first. The final entries per family are the
# fallbacks used when only the family name can be recognized.
DEFAULT_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec("claude-fable-5-1", "Claude Fable 5.1", "fable",
              r"(fable|mythos)-5-1\b",
              Rates(10.0, 50.0, cache_read_multiplier=0.025)),
    ModelSpec("claude-fable-5", "Claude Fable 5", "fable",
              r"(fable|mythos)-5\b|mythos-preview",
              Rates(10.0, 50.0)),
    ModelSpec("claude-opus-5", "Claude Opus 5", "opus",
              r"opus-5\b",
              Rates(5.0, 25.0), fast_rates=Rates(10.0, 50.0)),
    ModelSpec("claude-opus-4-8", "Claude Opus 4.8", "opus", r"opus-4-8\b", Rates(5.0, 25.0)),
    ModelSpec("claude-opus-4-7", "Claude Opus 4.7", "opus", r"opus-4-7\b", Rates(5.0, 25.0)),
    ModelSpec("claude-opus-4-6", "Claude Opus 4.6", "opus", r"opus-4-6\b", Rates(5.0, 25.0)),
    ModelSpec("claude-opus-4-5", "Claude Opus 4.5", "opus", r"opus-4-5\b", Rates(5.0, 25.0)),
    ModelSpec("claude-opus-4-1", "Claude Opus 4.1", "opus", r"opus-4-1\b", Rates(15.0, 75.0)),
    ModelSpec("claude-opus-4-0", "Claude Opus 4", "opus", r"opus-4(-0)?\b", Rates(15.0, 75.0)),
    ModelSpec("claude-3-opus", "Claude Opus 3", "opus", r"3-opus\b", Rates(15.0, 75.0)),
    ModelSpec("claude-sonnet-5", "Claude Sonnet 5", "sonnet", r"sonnet-5\b", Rates(2.0, 10.0)),
    ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", "sonnet", r"sonnet-4-6\b", Rates(3.0, 15.0)),
    ModelSpec("claude-sonnet-4-5", "Claude Sonnet 4.5", "sonnet", r"sonnet-4-5\b", Rates(3.0, 15.0)),
    ModelSpec("claude-sonnet-4-0", "Claude Sonnet 4", "sonnet", r"sonnet-4(-0)?\b", Rates(3.0, 15.0)),
    ModelSpec("claude-3-7-sonnet", "Claude Sonnet 3.7", "sonnet", r"3-7-sonnet\b", Rates(3.0, 15.0)),
    ModelSpec("claude-3-5-sonnet", "Claude Sonnet 3.5", "sonnet", r"3-5-sonnet\b", Rates(3.0, 15.0)),
    ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5", "haiku", r"haiku-4-5\b", Rates(1.0, 5.0)),
    ModelSpec("claude-3-5-haiku", "Claude Haiku 3.5", "haiku", r"3-5-haiku\b", Rates(0.80, 4.0)),
    ModelSpec("claude-3-haiku", "Claude Haiku 3", "haiku", r"3-haiku\b", Rates(0.25, 1.25)),
    # Family fallbacks: current-generation rates, flagged as estimates.
    ModelSpec("fable-family", "Fable (unrecognized version)", "fable", r"fable|mythos", Rates(10.0, 50.0)),
    ModelSpec("opus-family", "Opus (unrecognized version)", "opus", r"opus", Rates(5.0, 25.0)),
    ModelSpec("sonnet-family", "Sonnet (unrecognized version)", "sonnet", r"sonnet", Rates(2.0, 10.0)),
    ModelSpec("haiku-family", "Haiku (unrecognized version)", "haiku", r"haiku", Rates(1.0, 5.0)),
)

#: Used for model IDs that match nothing at all. Opus is what Claude Code
#: runs by default, so it is the least-surprising guess; the result is
#: still flagged as an estimate.
UNKNOWN_SPEC = ModelSpec("unknown", "Unknown model", "unknown", r"", Rates(5.0, 25.0))
SYNTHETIC_SPEC = ModelSpec("synthetic", "Synthetic (not billed)", "synthetic", r"", ZERO_RATES)

_DATE_SUFFIX = re.compile(r"-(20\d{6})$")
_VERTEX_VERSION = re.compile(r"@\d+$")
_PROVIDER_PREFIX = re.compile(r"^(?:[a-z]{2}\.)?anthropic\.")
_ROUTING_SUFFIX = re.compile(r"\[[^\]]*\]$")


def normalize_model_id(model: str) -> Tuple[str, bool]:
    """Strip deployment noise from a model ID.

    Returns the cleaned ID and whether it carried a ``-fast`` suffix (an
    older way of requesting fast mode that some deployments still emit).
    """
    m = (model or "").strip().lower()
    m = _PROVIDER_PREFIX.sub("", m)
    m = _ROUTING_SUFFIX.sub("", m)
    m = _VERTEX_VERSION.sub("", m)
    m = _DATE_SUFFIX.sub("", m)
    fast = False
    if m.endswith("-fast"):
        m = m[: -len("-fast")]
        fast = True
    return m, fast


@dataclass(frozen=True)
class Resolution:
    spec: ModelSpec
    rates: Rates
    #: ``exact`` for a known version, ``family`` when only the family was
    #: recognized, ``unknown`` for a total miss, ``synthetic`` for $0 lines.
    confidence: str
    fast: bool = False

    @property
    def is_estimate(self) -> bool:
        return self.confidence in ("family", "unknown")


class PricingTable:
    """Resolves model IDs to rates and prices usage records."""

    def __init__(self, models: Optional[List[ModelSpec]] = None):
        self._models: List[ModelSpec] = list(models or DEFAULT_MODELS)
        self._compiled = [(re.compile(m.pattern), m) for m in self._models if m.pattern]
        self._cache: Dict[Tuple[str, str], Resolution] = {}

    # -- lookup ----------------------------------------------------------

    @property
    def models(self) -> List[ModelSpec]:
        return list(self._models)

    def resolve(self, model: str, speed: str = "standard") -> Resolution:
        key = (model or "", speed or "standard")
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        if not model or model == SYNTHETIC_MODEL:
            res = Resolution(SYNTHETIC_SPEC, ZERO_RATES, "synthetic")
        else:
            normalized, suffix_fast = normalize_model_id(model)
            fast = suffix_fast or (speed == "fast")
            spec = UNKNOWN_SPEC
            confidence = "unknown"
            for pattern, candidate in self._compiled:
                if pattern.search(normalized):
                    spec = candidate
                    confidence = "family" if candidate.key.endswith("-family") else "exact"
                    break
            rates = spec.rates
            if fast and spec.fast_rates is not None:
                rates = spec.fast_rates
            res = Resolution(spec, rates, confidence, fast=fast)

        self._cache[key] = res
        return res

    def spec_for_key(self, key: str) -> Optional[ModelSpec]:
        for spec in self._models:
            if spec.key == key:
                return spec
        return None

    # -- pricing ---------------------------------------------------------

    @staticmethod
    def price_with_rates(usage: TokenUsage, rates: Rates) -> CostBreakdown:
        return CostBreakdown(
            input_cost=usage.input_tokens / MILLION * rates.input,
            output_cost=usage.output_tokens / MILLION * rates.output,
            cache_read_cost=usage.cache_read_tokens / MILLION * rates.cache_read,
            cache_write_cost=(
                usage.cache_write_5m_tokens / MILLION * rates.cache_write_5m
                + usage.cache_write_1h_tokens / MILLION * rates.cache_write_1h
            ),
            uncached_read_cost=usage.cache_read_tokens / MILLION * rates.input,
        )

    def price(self, usage: TokenUsage, model: str, speed: str = "standard") -> CostBreakdown:
        return self.price_with_rates(usage, self.resolve(model, speed).rates)

    # -- customization ---------------------------------------------------

    def with_overrides(self, overrides: Dict[str, dict]) -> "PricingTable":
        """Return a new table with rates replaced or models added.

        ``overrides`` maps a model key (``claude-opus-5``) or a brand-new
        key to a dict with ``input`` and ``output`` per-million rates and
        optional ``cache_read_multiplier`` / ``cache_write_5m_multiplier``
        / ``cache_write_1h_multiplier`` / ``pattern`` / ``display`` /
        ``family``. New models are inserted ahead of the family fallbacks.
        """
        by_key = {m.key: m for m in self._models}
        added: List[ModelSpec] = []
        for key, cfg in overrides.items():
            rate_kwargs = {k: float(cfg[k]) for k in (
                "input", "output", "cache_read_multiplier",
                "cache_write_5m_multiplier", "cache_write_1h_multiplier") if k in cfg}
            if key in by_key:
                existing = by_key[key]
                by_key[key] = replace(existing, rates=replace(existing.rates, **rate_kwargs))
            else:
                if "input" not in cfg or "output" not in cfg:
                    raise ValueError(f"override for new model '{key}' needs 'input' and 'output'")
                added.append(ModelSpec(
                    key=key,
                    display=str(cfg.get("display", key)),
                    family=str(cfg.get("family", "custom")),
                    pattern=str(cfg.get("pattern", re.escape(key))),
                    rates=Rates(**rate_kwargs),
                ))
        merged: List[ModelSpec] = []
        inserted = False
        for spec in self._models:
            if spec.key.endswith("-family") and not inserted:
                merged.extend(added)
                inserted = True
            merged.append(by_key[spec.key])
        if not inserted:
            merged.extend(added)
        return PricingTable(merged)

    @classmethod
    def from_json_file(cls, path: str) -> "PricingTable":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("pricing file must be a JSON object keyed by model")
        return cls().with_overrides(data)


DEFAULT_TABLE = PricingTable()
