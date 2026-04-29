"""Pytest fixtures for the Python torture suite.

Stage-3 status: skeleton. The `tiny_metal_nn` Python module does not exist
yet — it lands in stage 4 (`docs/know-how/006-python-binding-design.md` v2 §11).
Until then, the `tmnn` fixture below skips every test that depends on it.

When stage 4 lands, individual tests in `test_lifetime_torture.py` will
remove their `pytest.skip` markers one-by-one as the binding matures.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.fixture(scope="session")
def tmnn() -> Any:
    """Import `tiny_metal_nn`, skipping the entire test if the binding is
    not yet built.

    Stage 3: always skips (binding does not exist).
    Stage 4: skips only when build has not produced a wheel.
    """
    try:
        return importlib.import_module("tiny_metal_nn")
    except ImportError as exc:  # pragma: no cover — stage-3 path
        pytest.skip(
            f"tiny_metal_nn not importable ({exc}). "
            "The Python binding is a v1.0 deliverable; see "
            "docs/know-how/006-python-binding-design.md v2 §11 stage 4."
        )


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
