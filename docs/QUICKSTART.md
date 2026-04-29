# tiny-metal-nn Quickstart

`tiny-metal-nn` is the Metal-native tmnn core for hash-grid + MLP neural-field
training and inference on Apple GPUs.

`tmnn` is pre-1.0. The default standalone C++ train/eval path is supported, and
an optional Python binding ships alongside it. Both are walked through below.
Cross-device benchmark validation and additional flagship samples are still on
the roadmap — see [`STATUS.md`](../STATUS.md) for the current honest scope.

## Build

`tiny-metal-nn` now targets **C++23**. The public non-throwing factory surface
uses `tmnn::Result<T> = std::expected<T, DiagnosticInfo>`.

```bash
cmake -S . -B build -DTMNN_BUILD_SAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Headline workflow

The one-liner default path is:

```cpp
auto trainer = tmnn::create_trainer();
trainer.training_step(positions, targets, N);
trainer.inference(positions, output, N);
```

This path owns the default Metal runtime internally and is the recommended
starting point for standalone use.

If an evaluator-only consumer needs a separate `FieldEvaluator`, export one from
the trainer instead of constructing it directly from a model descriptor:

```cpp
auto trainer = tmnn::create_trainer();
auto evaluator = trainer.create_evaluator();
evaluator->evaluate(positions, output, N);
```

This keeps weight ownership honest: the evaluator is bound to the trainer's
runtime-owned buffers, not to an external descriptor-only model.

## Public sample

The repository ships a flagship runnable sample aligned with
tiny-cuda-nn's sample layout:

- `samples/mlp_learning_an_image.cpp` — config-driven C++ API, train a small
  RGB image model, then query it through inference

The one intentional API delta in that sample is input dimensionality: tmnn's
current default runtime requires `spatial_dims == 3 || 4`, so the sample lifts
image coordinates to `(x, y, 0)` rather than using pure 2D coordinates.

Build it explicitly with:

```bash
cmake --build build --target mlp_learning_an_image
```

Run it from the build tree:

```bash
./build/samples/mlp_learning_an_image
```

The sample now prints explicit `exit_code=...` on both success and failure
paths. Failure exits also print `error_stage=...`; inference/runtime failures
include the structured tmnn diagnostic string as well, and config/build failures
go through the non-throwing `try_create_from_config(...)` path instead of
landing as opaque exceptions. When Metal GPU execution is unavailable, the
sample exits non-zero instead of silently succeeding.

For the full diagnostics contract — `Result<T>`, `DiagnosticCode`, logger hook,
and the sample exit-code table — see `docs/ERROR-HANDLING.md`.

To inspect the realized runtime specialization on a successful run:

```bash
TMNN_SAMPLE_PRINT_RUNTIME_INSPECTION=1 ./build/samples/mlp_learning_an_image
```

## Python binding (optional)

The Python package wraps `Trainer` via pybind11. Install via the build
backend (scikit-build-core invokes the same CMake project):

```bash
python3.13 -m venv .venv
.venv/bin/pip install scikit-build-core pybind11
VCPKG_ROOT=$VCPKG_ROOT .venv/bin/pip install -e ".[dev]" --no-build-isolation
```

Headline workflow on the Python side:

```python
import tiny_metal_nn as tmnn
import numpy as np

config = {
    "encoding": {"otype": "HashGrid", "n_levels": 4, "log2_hashmap_size": 14},
    "network":  {"otype": "FullyFusedMLP", "n_neurons": 16, "n_hidden_layers": 1},
    "batch_size": 4096,
}
with tmnn.Trainer.from_config(config, n_input=3, n_output=1) as trainer:
    pos = np.random.randn(4096, 3).astype(np.float32)
    target = (np.linalg.norm(pos, axis=1, keepdims=True) - 0.5).astype(np.float32)
    for step in range(50):
        loss = trainer.training_step(pos, target)
    output = trainer.inference(pos)
```

Torch CPU tensors (`float32`, contiguous) are accepted directly by
`training_step` / `inference` via the numpy `__array__` protocol; non-CPU
tensors (MPS / CUDA) are rejected with an explicit `.cpu()` hint.

The full Python surface and design notes are in
[`ARCHITECTURE.md`](ARCHITECTURE.md) § 6.

## Migrating from tinycudann (Python)

If you have an existing `tinycudann`-based training script, the migration
tooling can do the mechanical conversion (imports + the canonical 5-line
training-loop body) and flag the rest:

```bash
.venv/bin/python tools/migrate_tcnn.py path/to/train.py --diff       # preview
.venv/bin/python tools/migrate_tcnn.py path/to/train.py --output train_tmnn.py
```

A worked example pair (the original tcnn version + the migrated tmnn
version + a CI convergence check) lives at
`examples/sphere_sdf/`. See
[`docs/TCNN-MIGRATION-GUIDE.md`](TCNN-MIGRATION-GUIDE.md) § 10 for the
field-by-field schema diff and the manual-edit catalog.

## Notes on current boundaries

- `Trainer` is the public standalone object for training and inference
- `create_from_config(...)` supports canonical `HashGrid` + `FullyFusedMLP` JSON
  plus `L2` / `L1` / `Huber` loss selection
- the current public sample deliberately follows the same "single flagship
  training sample" shape and near-matching file/target naming as tiny-cuda-nn
  rather than presenting a broad example matrix before the rest of the SDK
  surface is fully stabilized
