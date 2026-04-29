"""Stage-7 binding tests: torch CPU tensor I/O.

The v1.0 binding accepts PyTorch CPU tensors via the numpy array
protocol (zero-copy when the tensor is contiguous + float32). MPS /
CUDA / fp64 tensors are rejected explicitly with actionable error
messages — silent staging is anti-006-v2-§3.2.

MPS borrowed-input (zero-copy on the device) is on the v1.x+ roadmap
(006 v2 §12); v1.0 users move tensors to CPU explicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_training_step_accepts_torch_cpu_float32(tmnn, baseline_config):
    """Zero-copy via __array__ on CPU float32 tensors."""
    x = torch.randn(64, 3, dtype=torch.float32)
    y = torch.randn(64, 1, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        {**baseline_config, "batch_size": 64}, n_input=3, n_output=1
    ) as t:
        loss = t.training_step(x, y)
    assert math.isfinite(loss)


def test_training_step_rejects_torch_fp64(tmnn, baseline_config):
    """fp64 torch tensor raises TypeError with cast hint (006 v2 §3.2)."""
    x = torch.randn(64, 3, dtype=torch.float64)
    y = torch.randn(64, 1, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        with pytest.raises(TypeError) as exc:
            t.training_step(x, y)
    assert "float32" in str(exc.value)
    assert "float64" in str(exc.value)


def test_training_step_rejects_torch_mps_with_actionable_hint(
    tmnn, baseline_config
):
    """MPS tensor rejected; error message points at .cpu() and v1.x+ roadmap."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    x_mps = torch.randn(64, 3, dtype=torch.float32, device="mps")
    y = torch.randn(64, 1, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        with pytest.raises(ValueError) as exc:
            t.training_step(x_mps, y)
    msg = str(exc.value)
    assert "mps" in msg.lower()
    assert ".cpu()" in msg
    # 006 v2 §7.5: See: link must be a USER-VISIBLE doc, not a
    # gitignored know-how doc. TCNN-MIGRATION-GUIDE.md is in docs/.
    assert "TCNN-MIGRATION-GUIDE" in msg


def test_training_step_rejects_torch_non_contiguous(tmnn, baseline_config):
    """Non-contiguous torch slice → same hint as numpy non-contig."""
    big = torch.randn(64, 6, dtype=torch.float32)
    strided = big[:, :3]  # not contiguous
    y = torch.randn(64, 1, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        with pytest.raises(ValueError) as exc:
            t.training_step(strided, y)
    assert "contiguous" in str(exc.value)


def test_explicit_cpu_round_trip_works(tmnn, baseline_config):
    """User-driven .cpu() for an MPS tensor is the documented escape hatch."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    x_mps = torch.randn(64, 3, dtype=torch.float32, device="mps")
    y = torch.randn(64, 1, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        {**baseline_config, "batch_size": 64}, n_input=3, n_output=1
    ) as t:
        loss = t.training_step(x_mps.cpu(), y)
    assert math.isfinite(loss)


def test_inference_accepts_torch_input(tmnn, baseline_config):
    """Same coercion path as training_step. Returns a numpy array."""
    x = torch.randn(32, 3, dtype=torch.float32)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        out = t.inference(x)
    # Output is always numpy in v1.0 (006 v2 §3 / §12: torch wrapping is
    # a v1.x+ ergonomic improvement, not a correctness issue).
    assert isinstance(out, np.ndarray)
    assert out.shape == (32, 1)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_mixed_numpy_and_torch_inputs_work(tmnn, baseline_config):
    """input as torch CPU tensor, target as numpy (or vice versa) is OK."""
    x = torch.randn(64, 3, dtype=torch.float32)
    y = np.random.default_rng(0).standard_normal((64, 1)).astype(np.float32)
    with tmnn.Trainer.from_config(
        {**baseline_config, "batch_size": 64}, n_input=3, n_output=1
    ) as t:
        loss = t.training_step(x, y)
    assert math.isfinite(loss)
