"""Pytest fixtures for the Python tmnn test suite.

The `tmnn` fixture imports `tiny_metal_nn` and probes Metal GPU
availability once per session; tests that request it skip when no GPU
is present (e.g., a virtualized GitHub Actions macOS runner). Tests
that only check the module surface — no Trainer construction — should
import directly instead of requesting the fixture.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

import pytest


def _probe_gpu(module: Any) -> Optional[str]:
    """Try to construct a minimal Trainer. Return None on success or a
    short reason string on failure (used as the pytest.skip message).
    """
    try:
        with module.Trainer.from_config(
            {
                "encoding": {
                    "otype": "HashGrid",
                    "n_levels": 2,
                    "log2_hashmap_size": 10,
                },
                "network": {
                    "otype": "FullyFusedMLP",
                    "n_neurons": 16,
                    "n_hidden_layers": 1,
                },
            },
            n_input=3,
            n_output=1,
        ) as probe:
            if not probe.is_gpu_available():
                return "no Metal GPU available (e.g., virtualized CI macOS runner)"
    except Exception as exc:
        # Construction itself failed — almost always means no Metal device.
        return f"Trainer probe failed ({type(exc).__name__}: {exc})"
    return None


@pytest.fixture(scope="session")
def tmnn() -> Any:
    """Import `tiny_metal_nn` and probe GPU availability.

    Skips the test if the module is not importable (binding not built)
    OR if a minimal Trainer cannot be constructed (no Metal device).
    Tests that genuinely don't need a GPU — e.g., module-surface
    assertions — should import the module directly instead of using
    this fixture.
    """
    try:
        m = importlib.import_module("tiny_metal_nn")
    except ImportError as exc:
        pytest.skip(
            f"tiny_metal_nn not importable ({exc}). The Python binding "
            "is a v1.0 deliverable; build it first."
        )
    skip_reason = _probe_gpu(m)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    return m


@pytest.fixture
def baseline_config() -> dict:
    """Smallest config that should still drive a real training step.

    Mirrors `_baseline_config()` in 007 §6.1's reference test. Used by every
    torture test so the model construction itself is not the variable under
    test. Field names follow `011-json-schema-frozen.md` v1.0.
    """
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 4,
            "n_features_per_level": 2,
            "log2_hashmap_size": 14,
            "base_resolution": 16.0,
            "per_level_scale": 1.5,
        },
        "network": {
            "otype": "FullyFusedMLP",
            "n_neurons": 16,
            "n_hidden_layers": 1,
            "activation": "ReLU",
            "output_activation": "None",
        },
        "loss": {"otype": "L2"},
        "optimizer": {"otype": "Adam", "learning_rate": 1e-2},
    }
