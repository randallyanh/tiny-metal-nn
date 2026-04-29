"""CI convergence guard for `examples/`.

Per 006 v2 §4.3 / §11 stage 10: any PR that breaks one of the
migration examples should fail CI before merge. The cross-comparison
with the tcnn implementation needs CUDA hardware (not available on
Apple CI today, see the example's README); until then this test
locks in the tmnn-side convergence numbers measured at the time the
example was committed.

If a regression makes one of these numbers worse, do **not** raise the
threshold to make the test pass. Find the regression first.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Examples are not a Python package; add the example dir to the path
# so we can import its `train` module directly.
_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(_EXAMPLES_ROOT / "sphere_sdf" / "tmnn"))


@pytest.mark.slow
def test_sphere_sdf_tmnn_converges_below_threshold():
    """50-step sphere SDF training should land final loss < 0.01.

    Reference run on M1 Pro (2026-04-29):
        step  0: 0.130579
        step 10: 0.023629
        step 20: 0.007029
        step 30: 0.003719
        step 40: 0.001589
        step 49: 0.001447

    The 0.01 threshold is ~7× the observed final loss — generous so
    the test absorbs typical M-series variance, tight enough to catch
    a real regression (e.g., if Adam state were silently lost
    mid-training).
    """
    # The example imports tiny_metal_nn at module load, so the GPU
    # probe must run before that import. Mirror the conftest.py logic
    # in-line because this file does not request the `tmnn` fixture.
    import tiny_metal_nn as _tmnn  # noqa: PLC0415

    try:
        with _tmnn.Trainer.from_config(
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
        ) as _probe:
            if not _probe.is_gpu_available():
                pytest.skip("no Metal GPU available (CI runner virtualization?)")
    except Exception as exc:
        pytest.skip(f"Trainer probe failed ({type(exc).__name__}: {exc})")

    import train  # noqa: PLC0415

    losses = train.run(verbose=False)

    assert len(losses) == 50, f"expected 50 steps, got {len(losses)}"
    assert all(math.isfinite(loss) for loss in losses), (
        "non-finite loss observed during training"
    )
    final = losses[-1]
    assert final < 0.01, (
        f"sphere_sdf/tmnn final loss = {final:.6f}; expected < 0.01. "
        f"Either a regression in the binding / runtime, or the "
        f"reference threshold needs revisiting (find the cause first; "
        f"do not raise the threshold)."
    )
    # The loss must also have actually decreased — guards against a
    # bug where every step returns the same finite value.
    assert losses[0] > final * 5, (
        f"sphere_sdf/tmnn did not converge (start={losses[0]:.6f}, "
        f"end={final:.6f}). Optimizer is likely not stepping."
    )
