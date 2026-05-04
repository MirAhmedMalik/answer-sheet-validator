# =============================================================================
# tests/conftest.py — Pytest configuration and shared fixtures.
# Stubs heavy or unavailable modules (mlflow) BEFORE any test module imports
# grader.py, so the full test suite runs without needing mlflow installed.
# =============================================================================

import sys
import types
from unittest.mock import MagicMock


def _stub_mlflow() -> None:
    """
    Insert a minimal mlflow stub into sys.modules so grader.py can import it
    without needing the real mlflow (and its opentelemetry transitive dep)
    installed in the current environment.

    Only installed when the real mlflow is not available.
    """
    try:
        import mlflow  # noqa: F401 — if it exists, do nothing
    except ModuleNotFoundError:
        stub = types.ModuleType("mlflow")
        stub.set_experiment = MagicMock()
        stub.start_run = MagicMock()
        stub.log_param = MagicMock()
        stub.log_metric = MagicMock()

        # start_run() is used as a context manager
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        stub.start_run.return_value = ctx

        sys.modules["mlflow"] = stub


_stub_mlflow()
