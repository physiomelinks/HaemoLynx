"""Tests for pipeline module."""
import pytest

from ImageLynx.pipeline import run_pipeline


def test_run_pipeline_requires_skan():
    pytest.importorskip("skan")
    # Smoke test with minimal TIFF - would need a real file
    # Just verify the function is callable and has correct signature
    assert callable(run_pipeline)
    import inspect
    sig = inspect.signature(run_pipeline)
    assert "filepath" in sig.parameters
