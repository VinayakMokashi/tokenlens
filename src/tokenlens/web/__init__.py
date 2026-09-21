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

from ..analysis import AggregateReport, SessionReport, aggregate, analyze_session
from ..discovery import SessionRef, find_sessions, load_session, projects_root
from ..export import aggregate_to_dict, jsonable, session_report_to_dict
from ..findings import Finding, detect_findings, total_estimated_savings
from ..formatting import duration, money, pct, shorten, signed_money, tokens, when
from ..models import Session
from ..parser import iter_json_lines, parse_entries
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
    """In-memory cache of parsed sessions keyed by session ID."""

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

    def refresh(self) -> None:
        refs = find_sessions(self.root)
        with self.lock:
            seen = set()
            for ref in refs:
                sig = self._signature(ref)
                seen.add(ref.session_id)
                cached = self.entries.get(ref.session_id)
                if cached is not None and cached.signature == sig:
                    continue
                session = load_session(ref, self.pricing)
                self.entries[ref.session_id] = _Entry(
                    signature=sig,
                    session=session,
                    report=analyze_session(session, self.pricing),
                    findings=detect_findings(session, self.pricing),
                )
            for gone in set(self.entries) - seen:
                del self.entries[gone]

    def sessions(self) -> List[Session]:
        with self.lock:
            items = list(self.entries.values()) + list(self.uploaded.values())
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
        with self.lock:
            self.uploaded[session.session_id] = _Entry(
                signature=("upload", name),
                session=session,
                report=analyze_session(session, self.pricing),
                findings=detect_findings(session, self.pricing),
            )
        return session

    def all_findings(self) -> List[Finding]:
        with self.lock:
            items = list(self.entries.values()) + list(self.uploaded.values())
        findings: List[Finding] = []
        for e in items:
            findings.extend(e.findings)
        findings.sort(key=lambda f: (f.severity_rank, f.est_dollars_saved), reverse=True)
        return findings

    def aggregate(self, days: Optional[int] = None) -> AggregateReport:
        sessions = self.sessions()
        if days:
            from datetime import datetime, timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            sessions = [s for s in sessions if s.ended_at and s.ended_at >= cutoff]
        return aggregate(sessions, self.pricing)


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
        report = store().aggregate(days)
        findings = store().all_findings()
        chart = {
            "daily": [{"day": d.day.isoformat(), "cost": round(d.cost, 4), "turns": d.turns} for d in report.daily],
            "projects": [{"name": p.project_path.rstrip("/").split("/")[-1] or p.project_path,
                          "full": p.project_path, "cost": round(p.cost, 4)} for p in report.by_project[:10]],
            "models": [{"name": m.model, "cost": round(m.cost.total, 4)} for m in report.by_model if m.cost.total > 0],
        }
        return render_template(
            "dashboard.html", report=report, findings=findings[:12],
            finding_count=len(findings), avoidable=total_estimated_savings(findings),
            chart=chart, days=days, root=str(store().root), active="dashboard",
        )

    @app.route("/findings")
    def findings_page():
        findings = store().all_findings()
        total = sum(s.total_cost_with_subagents for s in store().sessions())
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
            "categories": [
                {"name": "Cache read (carrying context)", "cost": round(report.cost.cache_read_cost, 4)},
                {"name": "Cache write", "cost": round(report.cost.cache_write_cost, 4)},
                {"name": "Output (incl. thinking)", "cost": round(report.cost.output_cost, 4)},
                {"name": "Fresh input", "cost": round(report.cost.input_cost, 4)},
            ],
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
        return jsonify(aggregate_to_dict(store().aggregate(days), store().all_findings()))

    @app.route("/api/sessions")
    def api_sessions():
        report = store().aggregate()
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
