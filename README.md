# tokenlens

Understand what your Claude Code sessions cost, and why.

tokenlens reads the JSONL transcripts Claude Code writes under
`~/.claude/projects`, prices every API call at the model that actually ran
it, and explains where the money went: how the context window grew, which
tool results were expensive to carry, what compaction saved, and what the
same work would have cost on another model. It ships a dependency-free CLI
and an optional local web dashboard.

```text
$ tokenlens analyze latest

Session: Plan next phase for Information Verifier project
=========================================================
Project:   ~/projects/information-verifier
When:      2026-07-27 17:24 -> 2026-09-11 07:05   (active 5h 19m, wall 45d 13h)
Turns:     696 API calls for 34 prompts   subagents: 51   compactions: 1   API errors: 3

Cost
----
Total (this conversation):  $319.74
Subagents (51):             $113.07
Total including subagents:  $432.81

Category                        Tokens     Cost  Share
-----------------------------  -------  -------  -----
Cache read (carrying context)  282.19M  $170.52    53%
Cache write (1h)                 9.55M  $125.09
Cache write (all)                9.55M  $125.09    39%
Fresh input                       1.4K  $0.0089     0%
Output (incl. thinking)           759K   $24.13     8%

Carrying existing context:  $170.52 (53%)    New work:  $149.22
Cache hit rate:             96.7% of prompt tokens served from cache
Unit cost:                  $9.40 per prompt, $0.459 per API call

Context window
--------------
Peak 960K tokens, mean 419K per call.

Phase    Turns  Start ctx  Peak ctx  End ctx     Cost
-----  -------  ---------  --------  -------  -------
    1    1-558        37K      960K     960K  $295.73
    2  559-696        79K      326K     326K   $24.01
  compaction after turn 558 (auto): 970K -> 26K

Same conversation on another model (first-order estimate)
---------------------------------------------------------
Model               Cost  vs. actual
----------------  ------  --------------
Claude Fable 5.1  $299.47  -$20.27 (-6%)
Claude Opus 5     $255.55  -$64.19 (-20%)
Claude Sonnet 5   $102.22  -$217.52 (-68%)
Claude Haiku 4.5   $51.11  -$268.63 (-84%)

Findings
--------
Estimated avoidable spend: ~$105.27 of $319.74 (33%)

 1. [CRIT] 292 turns ran with context above 400K tokens  (~$105.27 avoidable)
      Turns 267-558 carried between 403K and 960K tokens of context. Re-
      reading that context from cache cost $117 across the run. Had the
      context been compacted to ~60K tokens at turn 267 (with /compact, or by
      starting a fresh session for the next task), roughly $105 of that would
      have been avoided.

 2. [info] Read pulled ~8K tokens into context  (~$2.18 avoidable)
      Turn 1: Read(PROJECT_PLAN.md) returned 30,197 characters. Those tokens
      were written to cache once and then re-read on each of the 557 turns
      that followed before the next compaction, for about $2.18 in total.
      Read a slice with offset/limit, or Grep for the lines you need.
```

## Why another token counter

Most analyzers total up tokens and multiply by a rate. That answers "how
much" but not "why", and in Claude Code the "why" is almost always the same
thing: **the whole conversation is re-sent on every API call.** A file you
read in turn 5 is paid for again, as a cache read, on turn 6, 7, 8 and every
turn after that until the context is compacted. In long sessions this carry
cost is over half the bill, and it is invisible in a token total.

tokenlens is built around that observation:

- Every turn records the **context size** the model received (fresh input +
  cache read + cache write) so you can see the window grow and reset.
- Cost is split into **carrying existing context** vs. **new work**.
- Findings price a tool result by how many turns it was carried, not as if
  it were seen once.
- Compaction events split the session into **phases** with their own cost.

## Features

- **Accurate pricing.** Per-turn rates for the model that ran that turn,
  including the Claude 5 family (Fable 5.1, Fable 5, Opus 5, Sonnet 5), the
  4.x generation, and legacy 3.x models. Cache writes are priced by TTL
  tier (1.25x input for 5-minute, 2x for 1-hour, which is what Claude Code
  mostly uses). Opus 5 fast mode is recognised. Unknown model IDs fall back
  to their family and are flagged, never silently mispriced. Override any
  rate with a JSON file.
- **Subagents included.** Transcripts spawned by the Agent tool and by the
  Workflow tool (`subagents/workflows/wf_*/`) are attached to their parent
  session. On a machine that uses workflows they can be a third of total
  spend.
- **Dollar-quantified findings.** Context bloat, duplicate file reads (a
  re-read after an edit is not flagged), large tool results, cache expiry
  after breaks, error streaks, heavy thinking, subagent share, and API
  errors, each with a plain-language explanation and a conservative
  saving estimate.
- **What-if pricing.** The same conversation re-priced on Fable 5.1, Opus 5,
  Sonnet 5, and Haiku 4.5.
- **Cross-session views.** Totals by project, model, and day; a ranked
  waste feed across every session; `--days N` windows.
- **Four output formats.** Human-readable text, versioned JSON, a Markdown
  write-up, and one CSV row per turn.
- **Web dashboard** (optional): context-growth charts with compaction
  markers, daily spend, findings, and a JSON API. Runs on localhost;
  nothing leaves your machine.
- **Robust on real data.** UTF-8 with replacement, malformed lines counted
  not fatal, Windows paths handled, project identity taken from the
  transcript's own `cwd` field instead of decoding directory names.
- **Zero dependencies** for the CLI. Python 3.9+.

## Install

```bash
pip install git+https://github.com/VinayakMokashi/tokenlens.git

# with the web dashboard
pip install "tokenlens[web] @ git+https://github.com/VinayakMokashi/tokenlens.git"
```

Or from a clone:

```bash
git clone https://github.com/VinayakMokashi/tokenlens.git
cd tokenlens
pip install -e ".[dev]"   # includes Flask and pytest
```

## Usage

```bash
tokenlens analyze latest                 # your most recent session
tokenlens analyze 2be1a3d1               # any session by ID prefix
tokenlens analyze path/to/session.jsonl  # any transcript file
tokenlens analyze latest --markdown      # shareable write-up
tokenlens analyze latest --json > s.json # everything, machine-readable
tokenlens analyze latest --csv turns.csv # one row per API call

tokenlens sessions                       # every local session, newest first
tokenlens sessions --project verifier    # filter by project path
tokenlens summary --days 30 --daily      # totals by project/model/day
tokenlens findings --min-dollars 1       # waste feed across sessions
tokenlens models                         # the pricing table in use
tokenlens web                            # dashboard at http://127.0.0.1:8765
```

Try it without your own data using the bundled synthetic transcript:

```bash
tokenlens analyze tests/fixtures/sample-session.jsonl
```

Transcripts are read from `~/.claude/projects` by default. Point elsewhere
with `--projects-dir DIR` or the `TOKENLENS_PROJECTS_DIR` environment
variable.

## How costs are computed

Anthropic bills four kinds of tokens on every call, and Claude Code records
all four in `message.usage`:

| Category | Rate (relative to input) | What it is in Claude Code |
|---|---|---|
| Fresh input | 1x | Almost nothing; the new user message |
| Cache write, 5-minute | 1.25x | Rare |
| Cache write, 1-hour | 2x | Everything new that enters context: tool results, your prompt, the model's own previous reply |
| Cache read | 0.1x (0.025x on Fable 5.1) | The entire conversation so far, re-sent every call |
| Output | 5x | The reply, including extended thinking |

Base rates per million tokens (Anthropic first-party API):

| Model | Input | Output |
|---|---:|---:|
| Claude Fable 5.1, Fable 5 | $10 | $50 |
| Claude Opus 5, 4.8, 4.7, 4.6, 4.5 | $5 | $25 |
| Claude Opus 4.1, 4 | $15 | $75 |
| Claude Sonnet 5 | $2 | $10 |
| Claude Sonnet 4.6, 4.5, 4, 3.7 | $3 | $15 |
| Claude Haiku 4.5 | $1 | $5 |

Run `tokenlens models` for the full table with cache and fast-mode rates.

**These are API-equivalent figures.** If you use Claude Code on a Pro or Max
subscription you pay a flat fee, not per token. The numbers still tell you
what the same usage would cost on the API, which sessions are expensive
relative to others, and how much of your plan's quota each habit consumes.

### Overriding rates

When a new model ships, or you are billed through a platform with different
prices, pass a JSON file:

```json
{
  "claude-opus-5": { "input": 5.0, "output": 25.0 },
  "claude-opus-6": {
    "input": 6.0, "output": 30.0,
    "pattern": "opus-6\\b", "display": "Claude Opus 6", "family": "opus",
    "cache_read_multiplier": 0.1, "cache_write_1h_multiplier": 2.0
  }
}
```

```bash
tokenlens --pricing rates.json analyze latest
```

Existing keys have their rates replaced; new keys are added ahead of the
family fallbacks.

## What the findings mean

| Rule | Severity | What it detects | Saving estimate |
|---|---|---|---|
| `context_bloat` | warning / critical | Runs of turns with context above 400K tokens | Cache reads above a 60K "healthy" context |
| `duplicate_read` | warning | A file read again with no Edit/Write to it in between | Write + carry cost of the duplicate |
| `large_tool_result` | info / warning | A single result over 16K characters | Write + carry cost until the next compaction |
| `cache_expired` | info | Zero cache reads after the first turn: the cache lapsed during a break | none (explained, not avoidable) |
| `model_switch` | info | Zero cache reads because the model changed; prompt caches are per model | none (explained) |
| `error_streak` | warning | Three or more consecutive failing tool calls | none (behavioural) |
| `thinking_share` | info | Extended thinking above 60% of output tokens, with effort levels used | none (points at the effort setting) |
| `subagent_share` | info | Subagents above 30% of session cost | none |
| `api_errors` | info | Rate limits and server errors (unbilled, but they stall work) | none |
| `estimated_pricing` | warning | A model ID not in the table; costs used family rates | none |

Thresholds live in `tokenlens.findings.Thresholds` and can be tuned when
calling the library directly. When a `context_bloat` finding exists, the
per-result savings whose carry window overlaps its turn range in the same
session are not added to the total, so the headline "estimated avoidable"
figure never double counts, while findings from other sessions or other
phases still count in full.

## Web dashboard

```bash
pip install "tokenlens[web]"
tokenlens web
```

The dashboard shows spend over time, by project and by model; a ranked
findings feed; and for each session a chart of context size per API call
with compaction markers, cost per call, phases, tools, subagents, and every
turn. Charts follow a few fixed rules: one axis per chart, thin marks,
hairline grids, a colorblind-safe palette, dark mode, and a table beside
every chart.

Parsed sessions are cached in memory and re-read only when a transcript
changes. A JSON API (`/api/summary`, `/api/sessions`, `/api/session/<id>`,
`/api/findings`) exposes the same data. You can also upload a transcript
from another machine; it is parsed in memory and never written to disk.

## Using tokenlens as a library

```python
from tokenlens.parser import parse_session
from tokenlens.analysis import analyze_session
from tokenlens.findings import detect_findings

session = parse_session("~/.claude/projects/<project>/<session>.jsonl")
report = analyze_session(session)
print(report.cost.total, report.carry_share, report.peak_context)
for finding in detect_findings(session):
    print(finding.severity, finding.title, finding.est_dollars_saved)
```

Modules, in dependency order:

| Module | Role |
|---|---|
| `models` | Dataclasses: `TokenUsage`, `CostBreakdown`, `Turn`, `Session`, ... |
| `pricing` | Model-ID resolution and rate cards |
| `parser` | JSONL transcript to `Session` |
| `discovery` | Find transcripts on disk, attach subagents |
| `analysis` | Context series, phases, what-if, aggregation |
| `findings` | Waste-detection rules |
| `formatting`, `report`, `export` | Text, JSON, Markdown, CSV rendering |
| `cli` | Command-line entry point |
| `web` | Flask dashboard (optional) |

## About the transcript format

Claude Code's transcript format is undocumented. The parser's module
docstring records what tokenlens relies on: how multi-line assistant
messages share a `message.id`, where cache-tier splits live, how compaction
boundaries and API errors are marked, and where subagent transcripts are
stored. If a future Claude Code release changes the layout, that docstring
is the place to start.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests build synthetic transcripts from helpers in `tests/conftest.py`; no
real session data is used or committed (`.gitignore` rejects `*.jsonl`
outside `tests/fixtures/`). CI runs the suite on Linux, macOS, and Windows
across Python 3.9, 3.12, and 3.13.

## Acknowledgements

tokenlens started as a rewrite of
[claude-token-analyzer](https://github.com/devraj-raghuvanshi/claude-token-analyzer)
by Devraj Raghuvanshi, whose root-cause approach to token waste inspired the
findings layer here.

## License

MIT. See [LICENSE](LICENSE).
