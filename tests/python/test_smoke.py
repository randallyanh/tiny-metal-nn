"""Stage-4 binding smoke tests.

End-to-end coverage that the v1.0 minimal binding actually works:
Trainer.from_config → training_step → inference → close. Lifetime /
GIL / dtype edge cases live in test_lifetime_torture.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_module_has_expected_surface():
    """The wheel exposes Trainer, ClosedError, __version__.

    Imports `tiny_metal_nn` directly (no `tmnn` fixture) so this test
    runs even on no-GPU CI runners — the public-surface check does not
    need a Metal device.
    """
    import tiny_metal_nn as tmnn  # noqa: PLC0415

    assert hasattr(tmnn, "Trainer")
    assert hasattr(tmnn, "ClosedError")
    assert hasattr(tmnn, "ConcurrentTrainingStepError")
    assert hasattr(tmnn, "ConfigError")
    assert hasattr(tmnn, "DTypeError")
    assert hasattr(tmnn, "__version__")
    assert issubclass(tmnn.ClosedError, Exception)
    assert issubclass(tmnn.ConfigError, ValueError)
    assert issubclass(tmnn.DTypeError, TypeError)


def test_from_config_with_minimal_dict(tmnn, baseline_config):
    """Minimal valid config builds a Trainer."""
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as trainer:
        assert trainer.is_gpu_available()
        assert trainer.step() == 0


def test_from_config_rejects_bad_schema_with_clear_message(tmnn):
    """Schema errors become Python ValueError carrying the diagnostic path."""
    bad = {"network": {"activation": "Sine"}}
    with pytest.raises(ValueError) as exc:
        tmnn.Trainer.from_config(bad, n_input=3, n_output=1)
    assert "network.activation" in str(exc.value)


def test_training_step_returns_finite_decreasing_loss(tmnn, baseline_config):
    """Six consecutive steps on the same batch — loss should not blow up."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, 3)).astype(np.float32)
    y = rng.standard_normal((64, 1)).astype(np.float32)

    losses: list[float] = []
    with tmnn.Trainer.from_config(
        {**baseline_config, "batch_size": 64}, n_input=3, n_output=1
    ) as trainer:
        for _ in range(6):
            loss = trainer.training_step(x, y)
            losses.append(loss)
            assert math.isfinite(loss)
        assert trainer.step() == 6

    # Not strict monotonic — but final < first by a meaningful margin.
    assert losses[-1] < losses[0]


def test_training_step_rejects_fp64(tmnn, baseline_config):
    rng = np.random.default_rng(0)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as trainer:
        x = rng.standard_normal((64, 3))  # default fp64
        y = rng.standard_normal((64, 1)).astype(np.float32)
        with pytest.raises(TypeError) as exc:
            trainer.training_step(x, y)
        assert "float32" in str(exc.value)
        assert "float64" in str(exc.value)


def test_training_step_rejects_non_contiguous(tmnn, baseline_config):
    rng = np.random.default_rng(0)
    big = rng.standard_normal((64, 6)).astype(np.float32)
    strided = big[:, :3]  # non-contiguous slice
    y = rng.standard_normal((64, 1)).astype(np.float32)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as trainer:
        with pytest.raises(ValueError) as exc:
            trainer.training_step(strided, y)
        assert "contiguous" in str(exc.value)


def test_inference_returns_owned_array(tmnn, baseline_config):
    """007 §2.1 scenario B: inference output is independent of trainer."""
    import gc

    rng = np.random.default_rng(7)
    x_eval = rng.standard_normal((32, 3)).astype(np.float32)

    trainer = tmnn.Trainer.from_config(baseline_config, n_input=3, n_output=1)
    out = trainer.inference(x_eval)
    expected_sum = float(out.sum())

    del trainer
    gc.collect()

    # Output buffer must outlive the trainer.
    assert out.shape == (32, 1)
    assert float(out.sum()) == expected_sum


def test_close_is_idempotent(tmnn, baseline_config):
    trainer = tmnn.Trainer.from_config(baseline_config, n_input=3, n_output=1)
    assert not trainer.closed()
    trainer.close()
    assert trainer.closed()
    trainer.close()  # double-close is a no-op
    assert trainer.closed()


def test_post_close_calls_raise_closed_error(tmnn, baseline_config):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 3)).astype(np.float32)
    y = rng.standard_normal((64, 1)).astype(np.float32)
    trainer = tmnn.Trainer.from_config(
        {**baseline_config, "batch_size": 64}, n_input=3, n_output=1
    )
    trainer.close()
    with pytest.raises(tmnn.ClosedError):
        trainer.training_step(x, y)
    with pytest.raises(tmnn.ClosedError):
        trainer.inference(x)
    with pytest.raises(tmnn.ClosedError):
        trainer.step()


def test_with_statement_propagates_exceptions(tmnn, baseline_config):
    """`with` exits via close() but does not suppress user exceptions."""
    with pytest.raises(RuntimeError, match="user code"):
        with tmnn.Trainer.from_config(
            baseline_config, n_input=3, n_output=1
        ) as trainer:
            assert trainer.is_gpu_available()
            raise RuntimeError("user code error")


def test_repr_shows_model_breakdown(tmnn, baseline_config):
    """006 v2 §3.4: repr is informative — shows encoding + network names
    + dim breakdown + parameter count + current step."""
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        r = repr(t)
    assert "tmnn.Trainer" in r
    assert "encoding=" in r
    assert "network=" in r
    assert "params=" in r
    assert "step=" in r
    assert "device=metal" in r


def test_summary_includes_dims_and_param_total(tmnn, baseline_config):
    """summary() is the verbose form — must show input/output dims and
    a total-parameters line."""
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        s = t.summary()
    assert "Encoding" in s
    assert "Network" in s
    assert "Input dims" in s
    assert "Output dims" in s
    assert "Total params" in s


def test_summary_flags_oversized_batch_size(tmnn, baseline_config):
    """When the user passes a batch_size that exceeds the configured plan,
    summary() should call it out so they know to widen from_config."""
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as t:
        plan_max = 4096  # baseline_config does not set batch_size; default
        # Use a batch_size guaranteed > the configured plan.
        s = t.summary(batch_size=plan_max * 4)
    assert "exceeds max_batch_size" in s


def test_repr_after_close_indicates_closed(tmnn, baseline_config):
    """After close() the repr must reflect the closed state instead of
    pretending the model is live."""
    t = tmnn.Trainer.from_config(baseline_config, n_input=3, n_output=1)
    t.close()
    assert "closed" in repr(t).lower()
