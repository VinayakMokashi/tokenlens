import json

import pytest

from tokenlens.models import TokenUsage
from tokenlens.pricing import (
    DEFAULT_TABLE,
    MILLION,
    PricingTable,
    normalize_model_id,
)


@pytest.mark.parametrize(
    "raw, expected_key, confidence",
    [
        ("claude-opus-5", "claude-opus-5", "exact"),
        ("claude-opus-5[1m]", "claude-opus-5", "exact"),
        ("claude-fable-5", "claude-fable-5", "exact"),
        ("claude-fable-5-1", "claude-fable-5-1", "exact"),
        ("claude-mythos-5-1", "claude-fable-5-1", "exact"),
        ("claude-sonnet-5", "claude-sonnet-5", "exact"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5", "exact"),
        ("anthropic.claude-opus-4-1-20250805", "claude-opus-4-1", "exact"),
        ("us.anthropic.claude-haiku-4-5-20251001", "claude-haiku-4-5", "exact"),
        ("claude-opus-4-5@20251101", "claude-opus-4-5", "exact"),
        ("claude-opus-4-20250514", "claude-opus-4-0", "exact"),
        ("claude-3-7-sonnet-20250219", "claude-3-7-sonnet", "exact"),
        ("claude-3-5-haiku-20241022", "claude-3-5-haiku", "exact"),
        ("claude-sonnet-7", "sonnet-family", "family"),
        ("claude-opus-9-2", "opus-family", "family"),
        ("some-other-vendor-model", "unknown", "unknown"),
        ("<synthetic>", "synthetic", "synthetic"),
        ("", "synthetic", "synthetic"),
    ],
)
def test_resolution(raw, expected_key, confidence):
    res = DEFAULT_TABLE.resolve(raw)
    assert res.spec.key == expected_key
    assert res.confidence == confidence
    assert res.is_estimate == (confidence in ("family", "unknown"))


def test_normalize_strips_fast_suffix():
    assert normalize_model_id("claude-opus-4-6-fast") == ("claude-opus-4-6", True)
    assert normalize_model_id("Claude-Opus-5[1m]") == ("claude-opus-5", False)


def test_current_generation_rates():
    opus = DEFAULT_TABLE.resolve("claude-opus-5").rates
    assert (opus.input, opus.output) == (5.0, 25.0)
    assert opus.cache_read == pytest.approx(0.5)
    assert opus.cache_write_5m == pytest.approx(6.25)
    assert opus.cache_write_1h == pytest.approx(10.0)

    fable = DEFAULT_TABLE.resolve("claude-fable-5").rates
    assert (fable.input, fable.output) == (10.0, 50.0)
    assert fable.cache_read == pytest.approx(1.0)

    fable51 = DEFAULT_TABLE.resolve("claude-fable-5-1").rates
    assert fable51.cache_read == pytest.approx(0.25)

    sonnet5 = DEFAULT_TABLE.resolve("claude-sonnet-5").rates
    assert (sonnet5.input, sonnet5.output) == (2.0, 10.0)

    haiku = DEFAULT_TABLE.resolve("claude-haiku-4-5").rates
    assert (haiku.input, haiku.output) == (1.0, 5.0)


def test_legacy_opus_rates_are_not_confused_with_current():
    assert DEFAULT_TABLE.resolve("claude-opus-4-1").rates.input == 15.0
    assert DEFAULT_TABLE.resolve("claude-opus-4-5").rates.input == 5.0


def test_fast_mode_uses_premium_rates_only_where_known():
    fast = DEFAULT_TABLE.resolve("claude-opus-5", speed="fast")
    assert fast.fast is True
    assert (fast.rates.input, fast.rates.output) == (10.0, 50.0)

    # Fast mode on a model without a known fast price falls back to
    # standard rates but still records that fast mode was requested.
    fallback = DEFAULT_TABLE.resolve("claude-sonnet-5", speed="fast")
    assert fallback.fast is True
    assert fallback.rates.input == 2.0


def test_price_separates_cache_write_tiers():
    usage = TokenUsage(
        input_tokens=1 * MILLION,
        output_tokens=1 * MILLION,
        cache_read_tokens=10 * MILLION,
        cache_write_5m_tokens=1 * MILLION,
        cache_write_1h_tokens=1 * MILLION,
    )
    cost = DEFAULT_TABLE.price(usage, "claude-opus-5")
    assert cost.input_cost == pytest.approx(5.0)
    assert cost.output_cost == pytest.approx(25.0)
    assert cost.cache_read_cost == pytest.approx(5.0)
    # 5m tier at 1.25x ($6.25) plus 1h tier at 2x ($10.00)
    assert cost.cache_write_cost == pytest.approx(16.25)
    assert cost.total == pytest.approx(51.25)
    # Reading 10M tokens from cache instead of paying input rate saved $45.
    assert cost.uncached_read_cost == pytest.approx(50.0)
    assert cost.cache_savings == pytest.approx(45.0)


def test_synthetic_lines_cost_nothing():
    usage = TokenUsage(input_tokens=5000, output_tokens=5000)
    assert DEFAULT_TABLE.price(usage, "<synthetic>").total == 0.0


def test_overrides_replace_existing_rates():
    table = PricingTable().with_overrides({"claude-opus-5": {"input": 1.0, "output": 2.0}})
    rates = table.resolve("claude-opus-5").rates
    assert (rates.input, rates.output) == (1.0, 2.0)
    # Multipliers are preserved when not overridden.
    assert rates.cache_write_1h == pytest.approx(2.0)
    # The default table is untouched.
    assert DEFAULT_TABLE.resolve("claude-opus-5").rates.input == 5.0


def test_overrides_add_new_models_ahead_of_family_fallbacks():
    table = PricingTable().with_overrides({
        "claude-opus-9": {"input": 7.0, "output": 35.0, "pattern": r"opus-9\b", "display": "Opus 9"},
    })
    res = table.resolve("claude-opus-9")
    assert res.spec.key == "claude-opus-9"
    assert res.confidence == "exact"
    assert res.rates.input == 7.0


def test_new_model_override_requires_both_rates():
    with pytest.raises(ValueError):
        PricingTable().with_overrides({"claude-new": {"input": 1.0}})


def test_from_json_file(tmp_path):
    p = tmp_path / "rates.json"
    p.write_text(json.dumps({"claude-haiku-4-5": {"input": 0.5, "output": 2.5}}), encoding="utf-8")
    table = PricingTable.from_json_file(str(p))
    assert table.resolve("claude-haiku-4-5").rates.output == 2.5
