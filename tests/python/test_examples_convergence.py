"""CI regression guards for `examples/`.

Each sample's reference numbers + thresholds live in its
`reference_metrics.json`. CI asserts the live run does not regress
beyond the recorded thresholds — final loss, per-step wall time, and
host RSS delta.

If a regression makes one of these numbers worse, do **not** raise the
threshold to make the test pass. Find the regression first.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import pytest

# Examples are not a Python package; add the example dir to the path
# so we can import its `train` module directly.
_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"
_SPHERE_SDF_DIR = _EXAMPLES_ROOT / "sphere_sdf"
sys.path.insert(0, str(_SPHERE_SDF_DIR / "tmnn"))


def _load_reference_metrics(sample_dir: Path) -> dict:
    """Read the sample's reference_metrics.json. Tests use only the
    `thresholds` block as gates; the other fields are documentation."""
    with (sample_dir / "reference_metrics.json").open() as f:
        return json.load(f)


@pytest.mark.slow
def test_sphere_sdf_tmnn_converges_and_meets_perf_thresholds():
    """End-to-end regression guard for examples/sphere_sdf/.

    Asserts:
      - all 50 steps return finite losses
      - final loss < `thresholds.final_loss_max` (~7× headroom over
        the 0.001447 reference)
      - convergence actually happened (loss[0] > 5 * loss[49])
      - median warm step time < `thresholds.step_warm_median_max_ms`
        (~3× headroom over the 1.4 ms reference; absorbs M-series
        variance without masking a real perf regression)
      - host RSS growth < `thresholds.host_rss_delta_max_mb` (~1.9×
        headroom; mostly catches GPU-buffer leaks since most of the
        baseline 188 MB is PyTorch + interpreter, which is constant)

    See `examples/sphere_sdf/reference_metrics.json` for the recorded
    machine + tmnn-version baseline.
    """
    metrics = _load_reference_metrics(_SPHERE_SDF_DIR)
    thresholds = metrics["thresholds"]

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

    rss_baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    import train  # noqa: PLC0415

    # Re-implement the train loop here so we can capture per-step times.
    # train.run() returns just the loss list; we want timings too.
    import torch
    import tiny_metal_nn as tmnn

    torch.manual_seed(42)
    losses: list[float] = []
    step_times_ms: list[float] = []
    with tmnn.Trainer.from_config(
        train.CONFIG, n_input=3, n_output=1
    ) as trainer:
        for _ in range(train.N_STEPS):
            positions = torch.rand(
                train.BATCH_SIZE, 3, dtype=torch.float32
            )
            target = (
                (positions - 0.5).norm(dim=1, keepdim=True)
                - train.SPHERE_RADIUS
            ).to(torch.float32)
            t0 = time.perf_counter()
            losses.append(trainer.training_step(positions, target))
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)

    rss_final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_rss_delta_mb = (rss_final - rss_baseline) / 1_000_000.0

    # ── convergence gates ──────────────────────────────────────────
    assert len(losses) == train.N_STEPS, (
        f"expected {train.N_STEPS} steps, got {len(losses)}"
    )
    assert all(math.isfinite(loss) for loss in losses), (
        "non-finite loss observed during training"
    )
    final = losses[-1]
    assert final < thresholds["final_loss_max"], (
        f"sphere_sdf final loss = {final:.6f}; threshold "
        f"{thresholds['final_loss_max']}. Find the regression first; "
        f"do not raise the threshold."
    )
    assert losses[0] > final * 5, (
        f"sphere_sdf did not converge (start={losses[0]:.6f}, "
        f"end={final:.6f}). Optimizer likely not stepping."
    )

    # ── perf gates ─────────────────────────────────────────────────
    # Drop step 0 (cold path: kernel-cache lookup / first commit).
    warm = sorted(step_times_ms[1:])
    median_warm_ms = warm[len(warm) // 2]
    assert median_warm_ms < thresholds["step_warm_median_max_ms"], (
        f"sphere_sdf step warm median = {median_warm_ms:.3f} ms; "
        f"threshold {thresholds['step_warm_median_max_ms']} ms. Some "
        f"hot-path kernel got slower; profile and find the cause."
    )

    assert host_rss_delta_mb < thresholds["host_rss_delta_max_mb"], (
        f"sphere_sdf host RSS grew by {host_rss_delta_mb:.1f} MB; "
        f"threshold {thresholds['host_rss_delta_max_mb']} MB. Likely "
        f"a leak in the binding / runtime — check buffer pool sizing "
        f"or shared_ptr cycles."
    )
