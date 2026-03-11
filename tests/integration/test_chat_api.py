"""Integration tests for MCP server mount (chat UI/API removed)."""

from __future__ import annotations

from mlb_predict.app.main import app


def test_mcp_mount_registered_in_app() -> None:
    """The FastAPI app has the /mcp mount when MCP is available.

    MCP creation is wrapped in try/except at module level, so the mount
    is only registered when create_mcp_app() succeeds. The test verifies
    that at most one mount exists and that the current environment has it.
    """
    from starlette.routing import Mount

    from mlb_predict.app.main import _mcp_app

    mounts = [r for r in app.routes if isinstance(r, Mount)]
    mcp_mounts = [m for m in mounts if m.path == "/mcp"]

    if _mcp_app is None:
        assert len(mcp_mounts) == 0, "MCP mount should not exist when MCP app failed to create"
    else:
        assert len(mcp_mounts) == 1, "Expected exactly one /mcp mount for MCP Streamable HTTP"
