"""Unit tests for MCP resilience — app starts even when MCP is unavailable.

Verifies that the FastAPI app can be created and serves routes when
create_mcp_app() fails or _mcp_app is None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    pass


class TestMCPModuleLevelResilience:
    """Tests for module-level MCP creation resilience."""

    def test_app_exists_and_has_routes(self) -> None:
        """The FastAPI app object is created regardless of MCP status."""
        from mlb_predict.app.main import app

        assert app is not None
        assert app.title == "MLB Win Probability"
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/health" in paths
        assert "/api/bootstrap-progress" in paths

    def test_mcp_app_is_not_none_in_normal_conditions(self) -> None:
        """Under normal conditions, _mcp_app should be created successfully."""
        from mlb_predict.app.main import _mcp_app

        assert _mcp_app is not None

    def test_bootstrap_progress_route_registered(self) -> None:
        """The /api/bootstrap-progress endpoint is always registered."""
        from mlb_predict.app.main import app

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/bootstrap-progress" in paths


class TestMCPMountConditional:
    """Tests for conditional /mcp mount behavior."""

    def test_mcp_mount_present_when_mcp_available(self) -> None:
        """When _mcp_app is not None, /mcp mount exists."""
        from starlette.routing import Mount

        from mlb_predict.app.main import _mcp_app, app

        if _mcp_app is None:
            pytest.skip("MCP not available in this environment")

        mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/mcp"]
        assert len(mounts) == 1

    def test_static_mount_always_present(self) -> None:
        """/static mount exists regardless of MCP status."""
        from starlette.routing import Mount

        from mlb_predict.app.main import app

        mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/static"]
        assert len(mounts) == 1


class TestCombinedLifespanResilience:
    """Tests for _combined_lifespan error handling."""

    def test_lifespan_function_exists(self) -> None:
        """The _combined_lifespan context manager is defined."""
        from mlb_predict.app.main import _combined_lifespan

        assert callable(_combined_lifespan)

    @pytest.mark.asyncio
    async def test_lifespan_handles_mcp_none(self) -> None:
        """_combined_lifespan yields even when _mcp_app is None.

        We patch _mcp_app to None and verify the lifespan still
        enters and exits cleanly.
        """
        from mlb_predict.app.main import _combined_lifespan, app

        with patch("mlb_predict.app.main._mcp_app", None):
            async with _combined_lifespan(app):
                pass
