"""Command-line interface.

    tokenlens analyze latest              # the most recent session
    tokenlens analyze 2be1a3d1            # a session by ID prefix
    tokenlens analyze path/to/file.jsonl  # any transcript file
    tokenlens sessions                    # every local session, newest first
    tokenlens summary --days 30           # totals by project, model, and day
    tokenlens findings                    # waste feed across all sessions
    tokenlens models                      # the pricing table in use
    tokenlens web                         # the dashboard (needs Flask)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .analysis import aggregate, analyze_session
from .discovery import find_sessions, load_sessions, projects_root, resolve_target
from .export import (
    aggregate_to_dict,
    render_markdown,
    session_report_to_dict,
    to_json,
    write_turns_csv,
)
from .findings import Finding, Thresholds, detect_findings
from .formatting import money, table
from .models import Session
from .parser import parse_session
from .pricing import DEFAULT_TABLE, PricingTable
from .report import render_daily, render_findings, render_session, render_sessions_table, render_summary


class CliError(Exception):
    """A user-facing error; printed without a traceback."""


# -- helpers -------------------------------------------------------------------


def _configure_stdout() -> None:
    """Make sure paths and titles with non-ASCII characters never crash the
    report on consoles with legacy code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _pricing(args: argparse.Namespace) -> PricingTable:
    if getattr(args, "pricing", None):
        try:
            return PricingTable.from_json_file(args.pricing)
        except (OSError, ValueError) as exc:
            raise CliError(f"could not load pricing overrides from {args.pricing}: {exc}")
    return DEFAULT_TABLE


def _root(args: argparse.Namespace) -> Path:
    return projects_root(getattr(args, "projects_dir", None))


def _filter_sessions(sessions: Sequence[Session], project: Optional[str], days: Optional[int]) -> List[Session]:
    result = list(sessions)
    if project:
        needle = project.lower()
        result = [s for s in result if needle in (s.project_path or s.project_dir).lower()]
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = [s for s in result if s.ended_at and s.ended_at >= cutoff]
    return result


def _all_findings(sessions: Sequence[Session], pricing: PricingTable,
                  thresholds: Optional[Thresholds] = None) -> List[Finding]:
    findings: List[Finding] = []
    for session in sessions:
        findings.extend(detect_findings(session, pricing, thresholds))
    findings.sort(key=lambda f: (f.severity_rank, f.est_dollars_saved), reverse=True)
    return findings


def _load_all(args: argparse.Namespace) -> List[Session]:
    pricing = _pricing(args)
    refs = find_sessions(_root(args))
    sessions = load_sessions(refs, pricing)
    return _filter_sessions(sessions, getattr(args, "project", None), getattr(args, "days", None))


# -- commands ------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    pricing = _pricing(args)
    try:
        path = resolve_target(args.target, _root(args))
    except FileNotFoundError as exc:
        raise CliError(str(exc))

    session = parse_session(str(path), pricing=pricing, include_subagents=not args.no_subagents)
    if not session.turns:
        raise CliError(f"{path} contains no billed API calls (is it a Claude Code transcript?)")

    report = analyze_session(session, pricing)
    findings = detect_findings(session, pricing)

    if args.csv:
        rows = write_turns_csv(session, args.csv)
        print(f"wrote {rows} turns to {args.csv}", file=sys.stderr)

    if args.json:
        print(to_json(session_report_to_dict(report, findings, include_turns=not args.no_turns)))
    elif args.markdown:
        print(render_markdown(report, findings))
    else:
        print(render_session(report, findings, findings_limit=None if args.all_findings else args.top))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    sessions = _load_all(args)
    if not sessions:
        raise CliError(f"no sessions found under {_root(args)}")
    report = aggregate(sessions, _pricing(args))
    if args.json:
        print(to_json(aggregate_to_dict(report, [])["sessions"][: args.limit]))
    else:
        print(render_sessions_table(report, limit=args.limit))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    pricing = _pricing(args)
    sessions = _load_all(args)
    if not sessions:
        raise CliError(f"no sessions found under {_root(args)}")
    report = aggregate(sessions, pricing)
    findings = _all_findings(sessions, pricing)
    if args.json:
        print(to_json(aggregate_to_dict(report, findings)))
        return 0
    label = f" (last {args.days} days)" if args.days else ""
    print(render_summary(report, findings, findings_limit=args.top, days_label=label))
    if args.daily:
        print()
        print(render_daily(report))
    return 0


def cmd_findings(args: argparse.Namespace) -> int:
    pricing = _pricing(args)
    sessions = _load_all(args)
    if not sessions:
        raise CliError(f"no sessions found under {_root(args)}")
    findings = _all_findings(sessions, pricing)
    if args.min_dollars:
        findings = [f for f in findings if f.est_dollars_saved >= args.min_dollars]
    if args.json:
        print(to_json([f.__dict__ for f in findings[: args.limit]]))
        return 0
    total = sum(s.total_cost_with_subagents for s in sessions)
    print(f"{len(findings)} findings across {len(sessions)} sessions ({money(total)} total spend)\n")
    print(render_findings(findings, limit=args.limit, show_provenance=True))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    pricing = _pricing(args)
    specs = [m for m in pricing.models if not m.key.endswith("-family")]
    if args.json:
        print(to_json([{"key": m.key, "display": m.display, "family": m.family,
                        "rates": m.rates.to_dict(),
                        "fast_rates": m.fast_rates.to_dict() if m.fast_rates else None} for m in specs]))
        return 0
    rows = []
    for m in specs:
        r = m.rates
        rows.append([m.key, m.display, f"{r.input:.2f}", f"{r.output:.2f}", f"{r.cache_read:.3f}",
                     f"{r.cache_write_5m:.2f}", f"{r.cache_write_1h:.2f}",
                     f"{m.fast_rates.input:.0f}/{m.fast_rates.output:.0f}" if m.fast_rates else ""])
    print("US dollars per million tokens (Anthropic first-party API rates)\n")
    print(table(["Model ID", "Name", "Input", "Output", "Cache read", "Write 5m", "Write 1h", "Fast in/out"], rows,
                align=["l", "l", "r", "r", "r", "r", "r", "r"]))
    print("\nUnrecognized IDs fall back to family rates (fable/opus/sonnet/haiku) and are flagged as estimates.")
    print("Override or add models with --pricing FILE (JSON keyed by model ID).")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    try:
        from .web import create_app
    except ImportError as exc:  # Flask missing
        raise CliError(
            "the web dashboard needs Flask: pip install 'tokenlens[web]'"
            f" ({exc})"
        )
    app = create_app(projects_dir=_root(args), pricing=_pricing(args))
    url = f"http://{args.host}:{args.port}/"
    print(f"tokenlens dashboard at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


# -- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenlens",
        description="Analyze Claude Code session transcripts: cost, context growth, waste, and what-ifs.",
        epilog=__doc__.split("\n\n", 1)[-1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"tokenlens {__version__}")
    parser.add_argument("--projects-dir", metavar="DIR",
                        help="transcript root (default: ~/.claude/projects or $TOKENLENS_PROJECTS_DIR)")
    parser.add_argument("--pricing", metavar="FILE",
                        help="JSON file overriding or adding per-model rates")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("analyze", help="analyze one session (file path, session ID prefix, or 'latest')")
    p.add_argument("target")
    p.add_argument("--json", action="store_true", help="emit the full report as JSON")
    p.add_argument("--markdown", action="store_true", help="emit a shareable Markdown write-up")
    p.add_argument("--csv", metavar="FILE", help="also write one CSV row per turn to FILE")
    p.add_argument("--top", type=int, default=10, metavar="N", help="findings to show (default 10)")
    p.add_argument("--all-findings", action="store_true", help="show every finding")
    p.add_argument("--no-subagents", action="store_true", help="ignore subagent transcripts")
    p.add_argument("--no-turns", action="store_true", help="omit per-turn rows from --json output")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("sessions", help="list local sessions, newest first")
    p.add_argument("--limit", type=int, default=50, metavar="N")
    p.add_argument("--project", metavar="TEXT", help="only projects whose path contains TEXT")
    p.add_argument("--days", type=int, metavar="N", help="only sessions active in the last N days")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("summary", help="totals across sessions by project, model, and day")
    p.add_argument("--days", type=int, metavar="N", help="only sessions active in the last N days")
    p.add_argument("--project", metavar="TEXT", help="only projects whose path contains TEXT")
    p.add_argument("--top", type=int, default=10, metavar="N", help="findings to show (default 10)")
    p.add_argument("--daily", action="store_true", help="append a per-day bar chart")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("findings", help="avoidable-spend findings across sessions, ranked")
    p.add_argument("--limit", type=int, default=25, metavar="N")
    p.add_argument("--min-dollars", type=float, default=0.0, metavar="X", help="hide findings below X dollars")
    p.add_argument("--project", metavar="TEXT")
    p.add_argument("--days", type=int, metavar="N")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_findings)

    p = sub.add_parser("models", help="show the pricing table in use")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("web", help="run the local web dashboard (requires Flask)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_web)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"tokenlens: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
