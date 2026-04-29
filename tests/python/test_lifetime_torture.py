"""Python torture suite — lifetime / GIL / leak / dtype safety.

Stage 3 status: skeleton (5 tests, all skipped). Each test is a placeholder
for a specific bug class identified in
`docs/know-how/007-python-binding-safety-engineering.md` § 6.1.

When the v1.0 Python binding lands (stage 4 of `006-python-binding-design.md`
v2 § 11), each test's `pytest.skip(...)` will be replaced by a real
implementation. Until then, every test reaches `pytest.skip` so CI confirms
the file is discoverable but no behavior is asserted.

Why these 5 (out of 12 in 007 § 6.1):
    1. test_trainer_destruction_no_leak    — host RSS leak / GPU memory leak
    2. test_gil_released_during_training_step — GIL not released → process freeze
    3. test_concurrent_training_step_raises   — concurrent calls undetected
    4. test_output_tensor_outlives_trainer    — output / trainer lifetime mismatch
    5. test_dtype_fp64_raises                 — fp64 silent cast → numerics bug

The other 7 tests in 007 § 6.1 (KeyboardInterrupt, ref-cycle, Metal memory
baseline, async inflight, non-contiguous input, long-run, close idempotent)
arrive during v1.0 polish phase (stage 11 in 006 v2 § 11).
"""

from __future__ import annotations

import gc
import math
import resource
import threading
import time

import numpy as np
import pytest


pytestmark = pytest.mark.torture


# ---------------------------------------------------------------------------
# 1. test_trainer_destruction_no_leak
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trainer_destruction_no_leak(tmnn, baseline_config):
    """N create/destroy cycles, host RSS high-water growth bounded.

    Bug class: a leak in the binding's lifetime chain (Python → C++ → Metal)
    surfaces here. Common culprits: shared_ptr cycles, unfinished GPU command
    buffers holding MTLBuffer references, missed dtor on Trainer.

    Stage 4.6: lands at 25 cycles (~1.5–2 minutes on M1 Pro) to catch
    per-cycle leaks within CI time. The full 1000-cycle target from 007
    § 6.1 is a v1.0 polish task once CI scheduling supports it.
    """
    cfg = dict(baseline_config)
    cfg["batch_size"] = 64
    rng = np.random.default_rng(0)
    input_arr = rng.standard_normal((64, 3)).astype(np.float32)
    target = rng.standard_normal((64, 1)).astype(np.float32)

    # Warm up so kernel-compilation high-water mark is captured before
    # baseline measurement; subsequent cycles must not accumulate.
    with tmnn.Trainer.from_config(cfg, n_input=3, n_output=1) as t:
        t.training_step(input_arr, target)
    gc.collect()

    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(25):
        with tmnn.Trainer.from_config(cfg, n_input=3, n_output=1) as t:
            t.training_step(input_arr, target)
        gc.collect()

    final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is bytes on macOS, KB on Linux. tmnn's CI matrix is
    # Apple Silicon (the only platform with a Metal device), so treat
    # the value as bytes.
    growth_bytes = final - baseline
    assert growth_bytes < 30_000_000, (
        f"host RSS grew by {growth_bytes / 1_000_000:.1f} MB after 25 "
        f"create/destroy cycles (threshold: 30 MB). Likely leak in "
        f"binding lifetime chain — check shared_ptr cycles, missed "
        f"unique_ptr resets, or Metal buffer retention."
    )


# ---------------------------------------------------------------------------
# 2. test_gil_released_during_training_step
# ---------------------------------------------------------------------------


def test_gil_released_during_training_step(tmnn, baseline_config):
    """Other thread can run sleep(1ms) while training_step is in-flight.

    Bug class: training_step holds the GIL → other Python threads (including
    the Ctrl-C handler, logger, dataloader prefetch) freeze. 007 § 1 covers
    the correct release-boundary pattern.
    """
    config = dict(baseline_config)
    # Bigger MLP so the training step is slow enough to be observable
    # against thread scheduling jitter.
    config["network"] = {**config["network"], "n_hidden_layers": 8,
                         "n_neurons": 64}
    config["batch_size"] = 8192
    rng = np.random.default_rng(0)
    input_arr = rng.standard_normal((8192, 3)).astype(np.float32)
    target = rng.standard_normal((8192, 1)).astype(np.float32)

    with tmnn.Trainer.from_config(config, n_input=3, n_output=1) as trainer:
        # Warm up so kernel-compilation cost is not on the measurement path.
        trainer.training_step(input_arr, target)

        other_woke_after: list[float] = []

        def other_thread():
            t0 = time.time()
            time.sleep(0.001)
            other_woke_after.append(time.time() - t0)

        th = threading.Thread(target=other_thread)
        th.start()
        trainer.training_step(input_arr, target)
        th.join()

    # If GIL were held during the GPU dispatch, the sleep would be deferred
    # well past 1 ms. 20 ms upper bound accommodates thread scheduling jitter.
    assert other_woke_after, "other thread did not record a wake time"
    assert other_woke_after[0] < 0.020, (
        f"GIL appears held: other thread slept for {other_woke_after[0] * 1000:.2f} ms "
        f"(threshold: 20 ms). Check the gil_scoped_release boundary in "
        f"src/python/tmnn_pybind.cpp."
    )


# ---------------------------------------------------------------------------
# 3. test_concurrent_training_step_raises
# ---------------------------------------------------------------------------


def test_concurrent_training_step_raises(tmnn, baseline_config):
    """Two threads calling the same trainer.training_step → at least one
    raises ConcurrentTrainingStepError (the other proceeds).

    Bug class: silent step-counter corruption from concurrent calls. Trainer
    is single-threaded by design (007 § 1.4); detection must be explicit.
    """
    config = dict(baseline_config)
    # Bigger MLP so each step takes long enough that two threads reliably
    # overlap inside training_step. Without this, thread A often finishes
    # before thread B reaches the atomic flag, hiding the bug under timing.
    config["network"] = {**config["network"], "n_hidden_layers": 8,
                         "n_neurons": 64}
    config["batch_size"] = 8192
    rng = np.random.default_rng(0)
    input_arr = rng.standard_normal((8192, 3)).astype(np.float32)
    target = rng.standard_normal((8192, 1)).astype(np.float32)

    with tmnn.Trainer.from_config(config, n_input=3, n_output=1) as trainer:
        # Warm up before measuring so kernel-compilation cost does not
        # mask the in-flight window.
        trainer.training_step(input_arr, target)

        # threading.Barrier so both threads enter training_step within the
        # same OS scheduler quantum; 5 iterations per thread give the race
        # multiple chances even if one trip happens to slip past.
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        successes: list[int] = []

        def worker(thread_idx: int):
            barrier.wait()
            for _ in range(5):
                try:
                    trainer.training_step(input_arr, target)
                    successes.append(thread_idx)
                except tmnn.ConcurrentTrainingStepError as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # At least one collision over 10 attempts must have raised the typed
    # exception. Both threads collectively must also still be doing real
    # work (not all calls colliding) — otherwise the guard might be
    # spuriously triggering.
    assert errors, (
        f"expected at least one ConcurrentTrainingStepError across 10 calls "
        f"from two threads (got 0 errors, {len(successes)} successes). "
        f"The atomic in_flight guard in src/python/tmnn_pybind.cpp may not "
        f"be wired correctly."
    )
    assert successes, "expected some calls to succeed (not all collided)"


def test_inference_and_training_step_share_concurrent_guard(tmnn, baseline_config):
    """training_step on thread A + inference on thread B must not race —
    they touch the same param store / GPU pool. PyTrainer.in_flight guards
    both call sites with the same atomic flag (007 §1.4 strict reading).
    """
    config = dict(baseline_config)
    config["network"] = {**config["network"], "n_hidden_layers": 8,
                         "n_neurons": 64}
    config["batch_size"] = 8192
    rng = np.random.default_rng(0)
    train_x = rng.standard_normal((8192, 3)).astype(np.float32)
    train_y = rng.standard_normal((8192, 1)).astype(np.float32)
    eval_x = rng.standard_normal((512, 3)).astype(np.float32)

    with tmnn.Trainer.from_config(config, n_input=3, n_output=1) as trainer:
        trainer.training_step(train_x, train_y)  # warm up

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def trainer_worker():
            barrier.wait()
            for _ in range(5):
                try:
                    trainer.training_step(train_x, train_y)
                except tmnn.ConcurrentTrainingStepError as e:
                    errors.append(e)

        def inference_worker():
            barrier.wait()
            for _ in range(5):
                try:
                    trainer.inference(eval_x)
                except tmnn.ConcurrentTrainingStepError as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=trainer_worker),
            threading.Thread(target=inference_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert errors, (
        "expected at least one ConcurrentTrainingStepError when one thread "
        "trains while another infers (10 attempts total). The PyTrainer "
        "in_flight guard must cover both call sites."
    )


# ---------------------------------------------------------------------------
# 4. test_output_tensor_outlives_trainer
# ---------------------------------------------------------------------------


def test_output_tensor_outlives_trainer(tmnn, baseline_config):
    """`output = trainer.inference(...); del trainer` — output stays valid.

    Bug class: inference returns a view into the trainer's pool, which gets
    reused / released when the trainer dies → reading `output` returns
    overwritten / freed memory. 007 § 2.1 scenario B requires inference to
    allocate a fresh output buffer that owns its lifetime.
    """
    rng = np.random.default_rng(0)
    trainer = tmnn.Trainer.from_config(baseline_config, n_input=3, n_output=1)
    input_arr = rng.standard_normal((64, 3)).astype(np.float32)
    output = trainer.inference(input_arr)
    expected_sum = float(output.sum())

    del trainer
    gc.collect()

    assert output.shape == (64, 1)
    assert math.isfinite(float(output.sum()))
    # Buffer integrity check: if the trainer's pool had been freed and the
    # view torn through, the sum would shift.
    assert float(output.sum()) == expected_sum


# ---------------------------------------------------------------------------
# 5. test_dtype_fp64_raises
# ---------------------------------------------------------------------------


def test_dtype_fp64_raises(tmnn, baseline_config):
    """fp64 input raises tmnn.DTypeError with an actionable message.

    Bug class: silent fp64 → fp32 cast loses precision and is the source of
    multiple nerfstudio-class numerical bugs. tmnn must reject explicitly.
    006 v2 § 3.2 / § 7.4 specify the exception type and message shape.

    v1.0 status: tmnn.DTypeError is currently aliased to the stdlib
    TypeError; a dedicated subclass ships in the stage-11 polish phase
    (006 v2 § 11). The alias keeps the user-visible name stable so this
    test will not regress when the subclass lands.
    """
    rng = np.random.default_rng(0)
    with tmnn.Trainer.from_config(
        baseline_config, n_input=3, n_output=1
    ) as trainer:
        input_arr = rng.standard_normal((64, 3))  # default fp64
        target = rng.standard_normal((64, 1)).astype(np.float32)

        with pytest.raises(tmnn.DTypeError) as exc:
            trainer.training_step(input_arr, target)

    assert "float64" in str(exc.value).lower()
    # 006 v2 §7.5 three-segment format: actionable Common: line points
    # at the explicit cast call. Match either numpy or torch path.
    assert "astype" in str(exc.value).lower() or "torch.tensor" in str(exc.value).lower()
