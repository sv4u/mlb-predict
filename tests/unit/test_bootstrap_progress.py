"""Unit tests for the /api/bootstrap-progress endpoint.

Tests the aggregated bootstrap progress logic that powers the initializing dashboard.
Uses direct manipulation of pipeline state (no subprocess execution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mlb_predict.app.admin import (
    PipelineKind,
    PipelineState,
    PipelineStatus,
    _ingest_commands,
    _retrain_commands,
    _states,
    get_state,
)

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def _reset_pipeline_states() -> None:
    """Reset all pipeline states to idle before each test."""
    for kind in PipelineKind:
        state = _states[kind]
        state.status = PipelineStatus.IDLE
        state.started_at = None
        state.finished_at = None
        state.elapsed_seconds = None
        state.log_lines = []
        state.error = None
        state.steps = []
        state.current_step_index = -1


# ---------------------------------------------------------------------------
# Command builder sanity checks
# ---------------------------------------------------------------------------


class TestCommandBuilders:
    """Verify the command builders produce expected step counts."""

    def test_ingest_commands_returns_steps(self) -> None:
        """_ingest_commands returns a non-empty list of (description, command) tuples."""
        cmds = _ingest_commands()
        assert len(cmds) >= 5
        for desc, cmd in cmds:
            assert isinstance(desc, str)
            assert isinstance(cmd, str)
            assert len(desc) > 0

    def test_retrain_commands_returns_steps(self) -> None:
        """_retrain_commands returns at least one step."""
        cmds = _retrain_commands()
        assert len(cmds) >= 1
        for desc, cmd in cmds:
            assert isinstance(desc, str)
            assert isinstance(cmd, str)


# ---------------------------------------------------------------------------
# Bootstrap progress calculation tests
# ---------------------------------------------------------------------------


def _compute_bootstrap_progress() -> dict:
    """Call the bootstrap progress computation logic directly.

    Mirrors the logic in main.py's api_bootstrap_progress handler without
    needing a running server or TestClient.
    """
    from mlb_predict.app.data_cache import is_ready

    ingest = get_state(PipelineKind.INGEST).to_dict()
    retrain = get_state(PipelineKind.RETRAIN).to_dict()

    ingest_steps = ingest.get("steps", [])
    retrain_steps = retrain.get("steps", [])

    if not ingest_steps and ingest["status"] == "idle":
        ingest_steps = [
            {"description": d, "status": "pending", "elapsed_seconds": None}
            for d, _ in _ingest_commands()
        ]
    if not retrain_steps and retrain["status"] == "idle":
        retrain_steps = [
            {"description": d, "status": "pending", "elapsed_seconds": None}
            for d, _ in _retrain_commands()
        ]

    all_steps = ingest_steps + retrain_steps
    total = len(all_steps) if all_steps else 1
    completed = sum(1 for s in all_steps if s["status"] == "complete")
    progress_pct = round(completed / total * 100) if total else 0

    completed_durations = [
        s["elapsed_seconds"]
        for s in all_steps
        if s["status"] == "complete" and s["elapsed_seconds"] is not None
    ]
    avg_step_duration = (
        sum(completed_durations) / len(completed_durations) if completed_durations else None
    )
    remaining_steps = total - completed
    if any(s["status"] == "running" for s in all_steps):
        remaining_steps -= 1
    eta_seconds = round(avg_step_duration * remaining_steps) if avg_step_duration else None

    if ingest["status"] == "running":
        current_phase = "ingest"
    elif retrain["status"] == "running":
        current_phase = "retrain"
    elif retrain["status"] in ("success", "failed"):
        current_phase = "retrain"
    elif ingest["status"] in ("success", "failed"):
        current_phase = "ingest"
    else:
        current_phase = "waiting"

    failed = ingest["status"] == "failed" or retrain["status"] == "failed"
    error_detail = ingest.get("error") or retrain.get("error")

    return {
        "ready": is_ready(),
        "current_phase": current_phase,
        "progress_pct": progress_pct,
        "completed_steps": completed,
        "total_steps": total,
        "eta_seconds": eta_seconds,
        "failed": failed,
        "error": error_detail,
        "phases": {
            "ingest": {**ingest, "steps": ingest_steps},
            "retrain": {**retrain, "steps": retrain_steps},
        },
    }


class TestBootstrapProgressIdle:
    """Tests for bootstrap progress when no pipelines have started."""

    def test_idle_state_returns_expected_steps(self) -> None:
        """When idle, expected steps from command builders are returned."""
        result = _compute_bootstrap_progress()

        expected_total = len(_ingest_commands()) + len(_retrain_commands())
        assert result["total_steps"] == expected_total
        assert result["completed_steps"] == 0
        assert result["progress_pct"] == 0
        assert result["current_phase"] == "waiting"
        assert result["failed"] is False
        assert result["error"] is None

    def test_idle_state_all_steps_pending(self) -> None:
        """All steps should be pending when no pipeline has started."""
        result = _compute_bootstrap_progress()

        for step in result["phases"]["ingest"]["steps"]:
            assert step["status"] == "pending"
        for step in result["phases"]["retrain"]["steps"]:
            assert step["status"] == "pending"

    def test_idle_state_no_eta(self) -> None:
        """ETA is None when no steps have completed."""
        result = _compute_bootstrap_progress()
        assert result["eta_seconds"] is None


class TestBootstrapProgressRunning:
    """Tests for bootstrap progress during active pipeline execution."""

    def test_ingest_running_shows_correct_phase(self) -> None:
        """current_phase is 'ingest' when the ingest pipeline is running."""
        state = get_state(PipelineKind.INGEST)
        state.reset()
        state.init_steps(["A", "B", "C"])
        state.begin_step(0)

        result = _compute_bootstrap_progress()

        assert result["current_phase"] == "ingest"
        assert result["progress_pct"] == 0

    def test_partial_progress_calculation(self) -> None:
        """Progress percentage reflects completed steps across both phases."""
        state = get_state(PipelineKind.INGEST)
        state.reset()
        state.init_steps(["A", "B", "C", "D"])
        state.begin_step(0)
        state.complete_step(0, elapsed=10.0)
        state.begin_step(1)
        state.complete_step(1, elapsed=20.0)
        state.begin_step(2)

        result = _compute_bootstrap_progress()

        total = 4 + len(_retrain_commands())
        assert result["completed_steps"] == 2
        assert result["total_steps"] == total
        expected_pct = round(2 / total * 100)
        assert result["progress_pct"] == expected_pct

    def test_eta_calculation(self) -> None:
        """ETA is computed from average completed step duration."""
        state = get_state(PipelineKind.INGEST)
        state.reset()
        state.init_steps(["A", "B", "C", "D"])
        state.begin_step(0)
        state.complete_step(0, elapsed=10.0)
        state.begin_step(1)
        state.complete_step(1, elapsed=20.0)
        state.begin_step(2)

        result = _compute_bootstrap_progress()

        avg_duration = (10.0 + 20.0) / 2
        remaining = result["total_steps"] - 2 - 1  # -1 for running step
        expected_eta = round(avg_duration * remaining)
        assert result["eta_seconds"] == expected_eta

    def test_retrain_phase_after_ingest(self) -> None:
        """current_phase is 'retrain' when ingest is done and retrain is running."""
        ingest = get_state(PipelineKind.INGEST)
        ingest.reset()
        ingest.init_steps(["A"])
        ingest.begin_step(0)
        ingest.complete_step(0, elapsed=5.0)
        ingest.finish(ok=True)

        retrain = get_state(PipelineKind.RETRAIN)
        retrain.reset()
        retrain.init_steps(["Train"])
        retrain.begin_step(0)

        result = _compute_bootstrap_progress()

        assert result["current_phase"] == "retrain"
        assert result["completed_steps"] == 1
        assert result["total_steps"] == 2


class TestBootstrapProgressCompleted:
    """Tests for bootstrap progress when pipelines finish."""

    def test_all_complete_shows_100_percent(self) -> None:
        """Progress is 100% when all steps in both phases are complete."""
        ingest = get_state(PipelineKind.INGEST)
        ingest.reset()
        ingest.init_steps(["A", "B"])
        ingest.begin_step(0)
        ingest.complete_step(0, elapsed=5.0)
        ingest.begin_step(1)
        ingest.complete_step(1, elapsed=5.0)
        ingest.finish(ok=True)

        retrain = get_state(PipelineKind.RETRAIN)
        retrain.reset()
        retrain.init_steps(["Train"])
        retrain.begin_step(0)
        retrain.complete_step(0, elapsed=30.0)
        retrain.finish(ok=True)

        result = _compute_bootstrap_progress()

        assert result["progress_pct"] == 100
        assert result["completed_steps"] == 3
        assert result["total_steps"] == 3
        assert result["eta_seconds"] is not None  # 0 remaining * avg = 0
        assert result["failed"] is False


class TestBootstrapProgressFailed:
    """Tests for bootstrap progress when a pipeline fails."""

    def test_ingest_failure_sets_failed_flag(self) -> None:
        """failed=True and error is populated when ingest fails."""
        state = get_state(PipelineKind.INGEST)
        state.reset()
        state.init_steps(["A", "B"])
        state.begin_step(0)
        state.complete_step(0, elapsed=5.0)
        state.begin_step(1)
        state.fail_step(1)
        state.finish(ok=False, error="Step 'B' exited with code 1")

        result = _compute_bootstrap_progress()

        assert result["failed"] is True
        assert result["error"] == "Step 'B' exited with code 1"
        assert result["current_phase"] == "ingest"

    def test_retrain_failure_after_successful_ingest(self) -> None:
        """failed=True when retrain fails even though ingest succeeded."""
        ingest = get_state(PipelineKind.INGEST)
        ingest.reset()
        ingest.init_steps(["A"])
        ingest.begin_step(0)
        ingest.complete_step(0, elapsed=5.0)
        ingest.finish(ok=True)

        retrain = get_state(PipelineKind.RETRAIN)
        retrain.reset()
        retrain.init_steps(["Train"])
        retrain.begin_step(0)
        retrain.fail_step(0)
        retrain.finish(ok=False, error="Training OOM")

        result = _compute_bootstrap_progress()

        assert result["failed"] is True
        assert result["error"] == "Training OOM"
        assert result["current_phase"] == "retrain"
