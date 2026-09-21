# Changelog

All notable changes to tokenlens are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-21

First release.

### Added

- Transcript parser for Claude Code JSONL sessions: folds multi-line
  assistant messages into one turn per API call, correlates tool calls with
  their results, records compaction events and API errors, reads the exact
  project path from `cwd`, and attaches subagent transcripts (including
  Workflow-spawned agents under `subagents/workflows/`).
- Pricing engine covering the Claude 5 family (Fable 5.1, Fable 5, Opus 5,
  Sonnet 5), the 4.x generation, and legacy 3.x models, with separate
  5-minute and 1-hour cache-write tiers, Opus 5 fast-mode rates, family
  fallbacks flagged as estimates, and JSON overrides via `--pricing`.
- Analysis: per-turn context series, carry cost vs. new work, phases between
  compactions, cache hit rate, usage by model and by tool, daily series,
  what-if re-pricing on four models, and cross-session aggregation.
- Findings: context bloat, duplicate reads (edit-aware), large tool results
  priced with their carry cost, cache expiry after breaks, error streaks,
  thinking share, subagent share, API errors, and estimated-pricing warnings.
- CLI: `analyze`, `sessions`, `summary`, `findings`, `models`, `web`, with
  text, JSON, Markdown, and CSV output.
- Optional Flask dashboard with context-growth charts, a daily spend chart,
  findings feed, JSON API, and in-memory upload.
- Test suite (131 tests) built on synthetic transcripts; CI on Linux, macOS,
  and Windows across Python 3.9, 3.12, and 3.13.

### Fixed (pre-release review)

- Version patterns such as `opus-4` no longer match later point releases
  (`claude-opus-4-9`); unknown versions fall through to the flagged family
  fallback instead of being priced as the base model with full confidence.
- Cache-write cost is tracked per TTL tier (`cache_write_5m_cost`,
  `cache_write_1h_cost`); the report's per-tier rows previously split one
  total by token share, which is wrong when both tiers appear.
- `total_estimated_savings` de-duplicates against context-bloat findings per
  session and per turn range, so one bloated session no longer hides the
  savings of every other session on the dashboard and in `summary --json`.
- Cache misses are explained but not counted as avoidable; a mid-session
  model switch (which also re-writes the cache) is reported as its own
  `model_switch` finding. The rule, now `cache_miss`, compares each call's
  cache read with the previous call's context instead of requiring zero
  reads, because Claude Code keeps a ~28K shared prefix cached even when the
  conversation's cache is gone; the old test missed every real case.
- Reading two different slices of one file is no longer flagged as a
  duplicate read.
- Thinking share is clamped at 100%.
- The dashboard's day filter now scopes findings and the avoidable KPI, not
  only the totals; rejected uploads are no longer kept as empty sessions;
  transcripts are parsed outside the store lock.
- Finding text uses the same number formatting as the rest of the report.
- Slash commands (`/clear`, `/model`) and their output are no longer counted
  as prompts or used as session titles, and transcripts with no billed calls
  are left out of listings.
- Local `<synthetic>` placeholder messages are no longer counted as turns.
- The dashboard and `summary` count subagent API calls alongside subagent
  spend, so the call count matches the model breakdown.
- Dashboard tables fit narrow windows and print without clipping.
- Workflow `journal.jsonl` files are no longer parsed as subagent
  transcripts; only `agent-*.jsonl` files are.
- A context-bloat run is no longer split in two by a single turn dipping
  just under the threshold.
- The session page shows the 1-hour and 5-minute cache-write costs.
