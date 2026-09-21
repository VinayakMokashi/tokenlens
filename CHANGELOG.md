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
- Test suite (115 tests) built on synthetic transcripts; CI on Linux, macOS,
  and Windows across Python 3.9, 3.12, and 3.13.
