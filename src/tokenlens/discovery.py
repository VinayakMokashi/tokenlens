"""Locate Claude Code transcripts on disk.

Layout under ``~/.claude/projects``::

    <encoded-project-dir>/<session-id>.jsonl
    <encoded-project-dir>/<session-id>/subagents/agent-<agent-id>.jsonl
    <encoded-project-dir>/<session-id>/subagents/workflows/wf_<id>/agent-<agent-id>.jsonl

The encoded directory name is the project's working directory with every
path separator, colon, space, and underscore replaced by ``-``. That is
not reversible (``my-app`` and ``my_app`` collide), so the parser reads
the real path from the transcript's ``cwd`` field and this module only
uses the directory name for grouping files that belong together.

Subagent transcripts outnumber main sessions by a wide margin (a single
code review can spawn a dozen), so they are attached to their parent
here rather than listed as sessions in their own right.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .models import Session
from .parser import parse_session
from .pricing import DEFAULT_TABLE, PricingTable

DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"


def projects_root(override: Optional[str] = None) -> Path:
    """Resolve the transcripts root: explicit argument, then the
    ``TOKENLENS_PROJECTS_DIR`` environment variable, then the default."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("TOKENLENS_PROJECTS_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_PROJECTS_ROOT


@dataclass
class SessionRef:
    """A transcript file found on disk, before parsing."""

    path: Path
    session_id: str
    project_dir: str
    mtime: float
    size: int
    is_subagent: bool = False
    parent_session_id: str = ""
    subagent_paths: List[Path] = field(default_factory=list)

    @property
    def latest_mtime(self) -> float:
        """Newest modification time across the session and its subagents."""
        latest = self.mtime
        for sub in self.subagent_paths:
            try:
                latest = max(latest, sub.stat().st_mtime)
            except OSError:
                continue
        return latest

    @property
    def total_size(self) -> int:
        total = self.size
        for sub in self.subagent_paths:
            try:
                total += sub.stat().st_size
            except OSError:
                continue
        return total


def find_sessions(root: Optional[Path] = None) -> List[SessionRef]:
    """Return every main session under ``root``, newest first, with
    subagent transcripts attached to their parents.

    Subagent files whose parent transcript is missing (rotated out or
    deleted) are returned as standalone refs flagged ``is_subagent`` so
    their cost is not silently lost.
    """
    base = root if root is not None else projects_root()
    if not base.is_dir():
        return []

    mains: Dict[str, SessionRef] = {}
    orphans: List[SessionRef] = []
    pending_subagents: List[tuple] = []

    for path in base.rglob("*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(base).parts
        if len(rel) < 2:
            continue
        project_dir = rel[0]

        if len(rel) == 2:
            ref = SessionRef(path=path, session_id=path.stem, project_dir=project_dir,
                             mtime=stat.st_mtime, size=stat.st_size)
            mains[_key(project_dir, ref.session_id)] = ref
        elif len(rel) >= 4 and rel[2] == "subagents":
            # Direct subagents sit right under subagents/; agents spawned by
            # the Workflow tool sit two levels deeper under workflows/wf_<id>/.
            parent_id = rel[1]
            pending_subagents.append((project_dir, parent_id, path, stat))
        # Anything else is an unknown layout; ignore rather than guess.

    for project_dir, parent_id, path, stat in pending_subagents:
        parent = mains.get(_key(project_dir, parent_id))
        if parent is not None:
            parent.subagent_paths.append(path)
        else:
            stem = path.stem
            orphans.append(SessionRef(
                path=path, session_id=stem[len("agent-"):] if stem.startswith("agent-") else stem,
                project_dir=project_dir, mtime=stat.st_mtime, size=stat.st_size,
                is_subagent=True, parent_session_id=parent_id,
            ))

    refs = list(mains.values()) + orphans
    for ref in refs:
        ref.subagent_paths.sort()
    refs.sort(key=lambda r: r.latest_mtime, reverse=True)
    return refs


def _key(project_dir: str, session_id: str) -> str:
    return f"{project_dir}/{session_id}"


def find_session(identifier: str, root: Optional[Path] = None) -> Optional[SessionRef]:
    """Find a session by full ID or unique ID prefix."""
    needle = identifier.strip().lower()
    if not needle:
        return None
    matches = [r for r in find_sessions(root) if r.session_id.lower().startswith(needle)]
    if len(matches) == 1:
        return matches[0]
    exact = [r for r in matches if r.session_id.lower() == needle]
    return exact[0] if len(exact) == 1 else None


def latest_session(root: Optional[Path] = None,
                   project_filter: Optional[Callable[[SessionRef], bool]] = None) -> Optional[SessionRef]:
    for ref in find_sessions(root):
        if ref.is_subagent:
            continue
        if project_filter is None or project_filter(ref):
            return ref
    return None


def resolve_target(target: str, root: Optional[Path] = None) -> Path:
    """Turn a CLI argument into a transcript path.

    Accepts an existing file path, the literal ``latest``, or a session
    ID / unique prefix. Raises ``FileNotFoundError`` with a helpful
    message otherwise.
    """
    candidate = Path(target).expanduser()
    if candidate.is_file():
        return candidate
    if target.lower() == "latest":
        ref = latest_session(root)
        if ref is None:
            raise FileNotFoundError(f"no sessions found under {root or projects_root()}")
        return ref.path
    ref = find_session(target, root)
    if ref is None:
        raise FileNotFoundError(
            f"'{target}' is not a file, 'latest', or a unique session ID prefix "
            f"under {root or projects_root()}"
        )
    return ref.path


def load_session(ref: SessionRef, pricing: PricingTable = DEFAULT_TABLE) -> Session:
    """Parse a ref, including its subagents when it is a main session."""
    session = parse_session(str(ref.path), pricing=pricing, include_subagents=not ref.is_subagent)
    if ref.is_subagent:
        session.is_subagent = True
        session.session_id = ref.session_id
        session.agent_id = session.agent_id or ref.session_id
        session.parent_session_id = session.parent_session_id or ref.parent_session_id
    session.project_dir = ref.project_dir
    return session


def load_sessions(refs: Iterable[SessionRef], pricing: PricingTable = DEFAULT_TABLE) -> List[Session]:
    return [load_session(ref, pricing) for ref in refs]
