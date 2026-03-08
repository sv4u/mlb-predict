"""Integration tests for chat page, chat API, and MCP server mount."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from winprob.app.main import app


def test_chat_page_returns_html() -> None:
    """GET /chat returns 200 and HTML."""
    client = TestClient(app)
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Chat" in resp.text
    assert "chat-messages" in resp.text


def test_chat_status_returns_dict() -> None:
    """GET /api/chat/status returns dict with ollama_available, model, session_count."""
    client = TestClient(app)
    resp = client.get("/api/chat/status")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "ollama_available" in data
    assert "model" in data
    assert "session_count" in data


def test_chat_post_returns_stream() -> None:
    """POST /api/chat with JSON body returns 200 and event-stream."""
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"message": "What seasons are available?", "session_id": "test"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    # Body may be empty or contain SSE lines
    text = resp.text
    if text:
        for line in text.split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                assert "content" in payload
                assert "done" in payload
                break


def test_mcp_mount_registered_in_app() -> None:
    """The FastAPI app has the /mcp mount (MCP server is registered)."""
    from starlette.routing import Mount

    mounts = [r for r in app.routes if isinstance(r, Mount)]
    mcp_mounts = [m for m in mounts if m.path == "/mcp"]
    assert len(mcp_mounts) == 1, "Expected exactly one /mcp mount for MCP Streamable HTTP"
