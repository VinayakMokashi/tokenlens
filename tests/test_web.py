import io
import json

import pytest

flask = pytest.importorskip("flask")

from conftest import SESSION_ID, assistant_line, build_sample_session, text_block, usage  # noqa: E402
from tokenlens.web import Store, create_app  # noqa: E402


@pytest.fixture
def client(sample_session_path):
    root = sample_session_path.parent.parent
    app = create_app(projects_dir=root)
    app.config["TESTING"] = True
    return app.test_client()


def test_dashboard_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "API-equivalent spend" in html
    assert "Fix failing test in app.py" in html
    assert 'id="daily-chart"' in html
    assert "Re-read an unchanged file" in html
    assert "Claude Haiku 4.5" in html  # what-if table


def test_dashboard_day_filter_scopes_findings_too(client):
    # The sample session is dated 2026-09-01, so a 1-day window excludes it.
    # Totals, findings, and the avoidable KPI must all describe the same
    # (empty) set of sessions.
    resp = client.get("/?days=1")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "No sessions found in the last 1 days" in html
    assert "Re-read an unchanged file" not in html

    summary = client.get("/api/summary?days=1").get_json()
    assert summary["session_count"] == 0
    assert summary["findings"] == []
    assert summary["estimated_avoidable"] == 0


def test_session_page_and_prefix_lookup(client):
    resp = client.get(f"/session/{SESSION_ID}")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Context window per API call" in html
    assert "Phases between compactions" in html
    assert "Compaction after turn 5" in html
    assert 'id="context-chart"' in html
    assert "All 6 API calls" in html

    assert client.get(f"/session/{SESSION_ID[:8]}").status_code == 200
    assert client.get("/session/does-not-exist").status_code == 404


def test_findings_page(client):
    resp = client.get("/findings")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "findings across all sessions" in html
    assert f"/session/{SESSION_ID}" in html


def test_json_api(client):
    summary = client.get("/api/summary").get_json()
    assert summary["session_count"] == 1
    assert summary["by_project"][0]["project_path"] == "C:/Users/Someone/Projects/demo-app"

    sessions = client.get("/api/sessions").get_json()
    assert sessions[0]["session_id"] == SESSION_ID

    detail = client.get(f"/api/session/{SESSION_ID}").get_json()
    assert len(detail["turns"]) == 6
    assert detail["metrics"]["peak_context"] == 16_705

    slim = client.get(f"/api/session/{SESSION_ID}?turns=0").get_json()
    assert "turns" not in slim

    findings = client.get("/api/findings").get_json()
    assert any(f["rule"] == "duplicate_read" for f in findings)

    assert client.get("/api/session/nope").status_code == 404
    assert client.get("/health").get_data(as_text=True) == "ok"


def test_upload_parses_in_memory_and_redirects(client):
    lines = "\n".join(json.dumps(e) for e in build_sample_session())
    data = {"file": (io.BytesIO(lines.encode("utf-8")), "uploaded-session.jsonl")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "/session/uploaded-session" in resp.headers["Location"]
    page = client.get(resp.headers["Location"])
    assert page.status_code == 200
    assert "uploaded-session" in page.get_data(as_text=True)


def test_upload_rejects_empty_and_missing_without_storing(client):
    assert client.post("/upload", data={}, content_type="multipart/form-data").status_code == 400
    data = {"file": (io.BytesIO(b'{"type": "queue-operation"}\n'), "junk-upload.jsonl")}
    assert client.post("/upload", data=data, content_type="multipart/form-data").status_code == 400
    # A rejected upload must not linger as an empty session.
    assert client.get("/session/junk-upload").status_code == 404
    assert all(s["session_id"] != "junk-upload" for s in client.get("/api/sessions").get_json())


def test_store_reuses_cache_until_file_changes(sample_session_path):
    root = sample_session_path.parent.parent
    store = Store(root=root)
    store.refresh()
    first = store.get(SESSION_ID)
    store.refresh()
    assert store.get(SESSION_ID) is first  # unchanged file: same entry object

    # Append a turn and bump mtime: the entry is rebuilt.
    extra = assistant_line("m_new", "2026-09-01T11:00:00Z", text_block("more"), usage(output_tokens=5))
    with sample_session_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")
    import os
    stat = sample_session_path.stat()
    os.utime(sample_session_path, (stat.st_atime + 10, stat.st_mtime + 10))
    store.refresh()
    rebuilt = store.get(SESSION_ID)
    assert rebuilt is not first
    assert len(rebuilt.session.turns) == 7


def test_refresh_endpoint_redirects(client):
    resp = client.post("/refresh", headers={"Referer": "/findings"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/findings")
