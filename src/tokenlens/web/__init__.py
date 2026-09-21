"""Local web dashboard (optional; requires Flask).

The dashboard reads the same transcripts the CLI does and renders them
with small interactive charts. Nothing leaves the machine: the server
binds to localhost and the only network requests the pages make are for
Chart.js from a CDN.

Parsed sessions are kept in memory and re-used while the underlying
files are unchanged, so navigating between pages is instant even with
hundreds of megabytes of transcripts.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..analysis import AggregateReport, SessionReport, aggregate, analyze_session, filter_sessions
from ..discovery import SessionRef, find_sessions, load_session, projects_root
from ..export import aggregate_to_dict, jsonable, session_report_to_dict
from ..findings import Finding, detect_findings, total_estimated_savings
from ..formatting import duration, money, pct, shorten, signed_money, tokens, when
from ..models import Session
from ..parser import parse_entries
from ..pricing import DEFAULT_TABLE, PricingTable

try:  # pragma: no cover - import guard exercised by the CLI
    from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for
except ImportError as exc:  # pragma: no cover
    raise ImportError("tokenlens.web requires Flask; install with: pip install 'tokenlens[web]'") from exc


@dataclass
class _Entry:
    signature: Tuple
    session: Session
    report: SessionReport
    findings: List[Finding]


@dataclass
class Store:
    """In-memory cache of parsed sessions keyed by session ID.

    Parsing happens outside the lock so a slow refresh (a freshly written
    200 MB transcript) never blocks other requests; the lock only guards
    the dictionary swap.
    """

    root: Path
    pricing: PricingTable = DEFAULT_TABLE
    entries: Dict[str, _Entry] = field(default_factory=dict)
    uploaded: Dict[str, _Entry] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _signature(ref: SessionRef) -> Tuple:
        subs = []
        for p in ref.subagent_paths:
            try:
                st = p.stat()
                subs.append((p.name, st.st_mtime, st.st_size))
            except OSError:
                continue
        return (str(ref.path), ref.mtime, ref.size, tuple(subs))

    def _build(self, session: Session, signature: Tuple) -> _Entry:
        return _Entry(
            signature=signature,
            session=session,
            report=analyze_session(session, self.pricing),
            findings=detect_findings(session, self.pricing),
        )

    def refresh(self) -> None:
        refs = find_sessions(self.root)
        with self.lock:
            current = dict(self.entries)

        stale: List[Tuple[SessionRef, Tuple]] = []
        seen = set()
        for ref in refs:
            sig = self._signature(ref)
            seen.add(ref.session_id)
            cached = current.get(ref.session_id)
            if cached is None or cached.signature != sig:
                stale.append((ref, sig))

        rebuilt = {ref.session_id: self._build(load_session(ref, self.pricing), sig) for ref, sig in stale}

        with self.lock:
            for gone in set(self.entries) - seen:
                del self.entries[gone]
            self.entries.update(rebuilt)

    def _all_entries(self) -> List[_Entry]:
        with self.lock:
            return list(self.entries.values()) + list(self.uploaded.values())

    def sessions(self) -> List[Session]:
        items = self._all_entries()
        items.sort(key=lambda e: (e.session.ended_at.timestamp() if e.session.ended_at else 0), reverse=True)
        return [e.session for e in items]

    def get(self, session_id: str) -> Optional[_Entry]:
        with self.lock:
            entry = self.entries.get(session_id) or self.uploaded.get(session_id)
            if entry is None:
                matches = [e for k, e in list(self.entries.items()) + list(self.uploaded.items())
                           if k.startswith(session_id)]
                entry = matches[0] if len(matches) == 1 else None
        return entry

    def add_uploaded(self, name: str, lines: Sequence[str]) -> Session:
        """Parse an uploaded transcript in memory. It is only kept when it
        contains billed API calls, so a rejected upload leaves no trace."""
        entries = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError:
                entries.append(None)
        session = parse_entries(entries, path=name, pricing=self.pricing)
        session.session_id = session.session_id or Path(name).stem
        if not session.title:
            session.title = Path(name).name
        if session.turns:
            entry = self._build(session, ("upload", name))
            with self.lock:
                self.uploaded[session.session_id] = entry
        return session

    def findings_for(self, sessions: Sequence[Session]) -> List[Finding]:
        wanted = {s.session_id for s in sessions}
        findings: List[Finding] = []
        for e in self._all_entries():
            if e.session.session_id in wanted:
                findings.extend(e.findings)
        findings.sort(key=lambda f: (f.severity_rank, f.est_dollars_saved), reverse=True)
        return findings

    def all_findings(self) -> List[Finding]:
        return self.findings_for(self.sessions())

    def window(self, days: Optional[int] = None) -> Tuple[List[Session], AggregateReport, List[Finding]]:
        """Sessions, aggregate, and findings for one time window, so every
        widget on a page describes the same set of sessions."""
        sessions = filter_sessions(self.sessions(), days=days)
        return sessions, aggregate(sessions, self.pricing), self.findings_for(sessions)


def create_app(projects_dir: Optional[Path] = None, pricing: PricingTable = DEFAULT_TABLE,
               store: Optional[Store] = None) -> "Flask":
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["STORE"] = store or Store(root=projects_dir or projects_root(), pricing=pricing)

    app.jinja_env.filters.update({
        "money": money,
        "signed_money": signed_money,
        "tokens": tokens,
        "pct": pct,
        "when": when,
        "duration": duration,
        "shorten": shorten,
    })

    def store() -> Store:
        return app.config["STORE"]

    @app.before_request
    def _refresh_on_page_load() -> None:
        if request.endpoint in ("dashboard", "findings_page", "session_page", "api_summary",
                                "api_sessions", "api_findings"):
            store().refresh()

    # -- pages -----------------------------------------------------------

    @app.route("/")
    def dashboard():
        days = request.args.get("days", type=int)
        _sessions, report, findings = store().window(days)
        chart = {
            "daily": [{"day": d.day.isoformat(), "cost": round(d.cost, 4), "turns": d.turns} for d in report.daily],
        }
        return render_template(
            "dashboard.html", report=report, findings=findings[:12],
            finding_count=len(findings), avoidable=total_estimated_savings(findings),
            chart=chart, days=days, root=str(store().root), active="dashboard",
        )

    @app.route("/findings")
    def findings_page():
        sessions = store().sessions()
        findings = store().findings_for(sessions)
        total = sum(s.total_cost_with_subagents for s in sessions)
        return render_template("findings.html", findings=findings, total=total,
                               avoidable=total_estimated_savings(findings), active="findings")

    @app.route("/session/<session_id>")
    def session_page(session_id: str):
        entry = store().get(session_id)
        if entry is None:
            abort(404)
        report, findings, session = entry.report, entry.findings, entry.session
        chart = {
            "context": [{"turn": p.turn_index, "context": p.context_tokens, "cost": round(p.cost, 5),
                         "carry": round(p.carry_cost, 5), "compaction": p.compaction_before,
                         "model": p.model} for p in report.context],
        }
        return render_template(
            "session.html", s=session, report=report, findings=findings,
            avoidable=total_estimated_savings(findings), chart=chart,
            turns=session.turns, active="sessions",
        )

    @app.route("/upload", methods=["POST"])
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            abort(400, "no file provided")
        text = file.read().decode("utf-8", errors="replace")
        session = store().add_uploaded(file.filename, text.splitlines())
        if not session.turns:
            abort(400, "that file contains no billed API calls")
        return redirect(url_for("session_page", session_id=session.session_id))

    @app.route("/refresh", methods=["POST"])
    def refresh():
        store().refresh()
        return redirect(request.referrer or url_for("dashboard"))

    # -- JSON API --------------------------------------------------------

    @app.route("/api/summary")
    def api_summary():
        days = request.args.get("days", type=int)
        _sessions, report, findings = store().window(days)
        return jsonify(aggregate_to_dict(report, findings))

    @app.route("/api/sessions")
    def api_sessions():
        _sessions, report, _findings = store().window(None)
        return jsonify(jsonable(report.sessions))

    @app.route("/api/session/<session_id>")
    def api_session(session_id: str):
        entry = store().get(session_id)
        if entry is None:
            abort(404)
        include_turns = request.args.get("turns", "1") != "0"
        return jsonify(session_report_to_dict(entry.report, entry.findings, include_turns=include_turns))

    @app.route("/api/findings")
    def api_findings():
        return jsonify(jsonable(store().all_findings()))

    @app.route("/health")
    def health():
        return Response("ok", mimetype="text/plain")

    return app
