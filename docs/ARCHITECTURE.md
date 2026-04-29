# tiny-metal-nn Architecture

A short architectural overview of the `tiny-metal-nn` (`tmnn`) C++ library:
what the library is composed of, how a training/inference call flows through
it, and where the boundaries between layers sit.

For a comparison against `tiny-cuda-nn` and `MLX`, see
[`docs/VS-MLX-AND-TCNN.md`](VS-MLX-AND-TCNN.md). For what the library does and
does not currently do, see [`STATUS.md`](../STATUS.md).

---

## 1. Library composition

`tmnn` ships as four CMake targets, each with a distinct role:

| Target | Kind | Role |
|---|---|---|
| `tiny_metal_nn_core` | INTERFACE (header-only) | Public API: `Encoding`, `Network`, `NetworkWithInputEncoding`, `Trainer`, `Loss`, `Optimizer`, `Result<T>` / `DiagnosticInfo`, JSON config types |
| `tiny_metal_nn_kernels` | STATIC | Metal Shading Language (MSL) kernel generation: `KernelSpec`, `KernelCompiler`, `MLPKernelEmitter` |
| `tiny_metal_nn_runtime` | STATIC | Metal device runtime: `MetalContext`, pipeline registry, batch pool, training-step lifecycle, autotune manifest, numerics guard |
| `tiny_metal_nn_extensions` | STATIC | Built-in training adapters: DNL, RMHE, 4D, standard SDF, multi-output MLP |
| `_C` (pybind11 module) | SHARED (Python module) | Python binding: `tiny_metal_nn.Trainer`, typed exceptions, numpy / torch CPU tensor I/O. Built when `-DTMNN_BUILD_PYTHON_MODULE=ON` |

Public API headers live in `include/tiny-metal-nn/`. Implementation lives in
`src/tiny_metal_nn/`. Adapters and runtime internals are not part of the
header-only core surface.

The library installs as a CMake package; downstream consumers can use:

```cmake
find_package(tiny_metal_nn CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tiny_metal_nn::tiny_metal_nn_runtime)
```

---

## 2. Call flow: from JSON config to a fused training step

```
JSON config dict
   │
   ▼  factory_json::canonicalize_model_config
canonical encoding/network/loss/optimizer descriptors
   │
   ▼  create_encoding / create_network / create_network_with_input_encoding
NetworkWithInputEncoding (Encoding + Network composition)
   │
   ▼  NetworkPlan::plan(NetworkFactoryOptions)
selected execution family (FullyFusedMetal / TiledMetal / SafeDebugMetal),
candidate families, planner reasons
   │
   ▼  Trainer construction (binds runtime + loss + optimizer)
Trainer<Module, Loss, Optimizer, ITrainerRuntime>
   │
   ▼  trainer.training_step(input, target, N)
ITrainerRuntime::training_step
   ├─ MetalContext: pipeline registry / batch pool / step lane coordinator
   ├─ MLPKernelEmitter → MSL string → KernelCompiler → MTLComputePipelineState
   └─ fused forward + backward + Adam dispatch on Metal command buffer
   │
   ▼
TrainingStepResult { step, loss, numerics report, optional probe }
```

Inference shares the runtime authority but skips the loss/backward path:

```
trainer.inference(positions, output, N)
   │
   ▼  ITrainerRuntime::inference (forced family or planner-selected)
   │
   ▼  evaluate kernel (FullyFusedMetal or TiledMetal pipeline)
   │
   ▼ output buffer
```

---

## 3. Public surface boundaries

The library distinguishes four ownership tiers:

1. **Header-only public types** — `Encoding`, `Network`, `Trainer`,
   `Loss`, `Optimizer`, `Result<T>`. These are the API users build against.
2. **Public runtime types** — `MetalContext`, `NetworkPlan`,
   `OptimizerStateBlob`, `TrainingStepResult`. Stable but tied to the Metal
   runtime; users who want fused training and inspection use them directly.
3. **Extension SDK** — `TrainingAdapter` and adapter-side schema/lowering
   types. Users who add new training shapes (custom encoding + loss + batch
   layout combinations) implement against this seam. See
   [`docs/EXTENSIBILITY-DESIGN.md`](EXTENSIBILITY-DESIGN.md).
4. **`detail/` headers** — implementation organization for the default
   runtime. Types reachable through the public headers above (e.g.
   `OptimizerStateBlob`, returned by `Trainer::export_optimizer_state()`)
   are part of the public surface and are versioned via their own contracts.
   Types and helpers that are only used internally by the runtime are
   subject to change without notice.

---

## 4. Design choices

A handful of choices distinguish `tmnn` from a generic neural-network library:

- **Fused over composable.** `tmnn`'s training kernels fuse forward, backward,
  and (when applicable) the optimizer step into a single Metal dispatch.
  Users who need PyTorch-style operator composition should use a tensor
  framework such as [MLX](https://github.com/ml-explore/mlx); `tmnn` trades
  composability for fused-kernel performance.
- **Explicit family selection.** `NetworkPlan` records which execution family
  was selected, the candidates considered, and the reason for the choice.
  Users who want reproducible runtime behavior can inspect and act on that
  decision instead of relying on hidden heuristics.
- **Manifest-backed kernel prewarm.** Compiled kernel specializations are
  recorded in an `AutotuneManifest` on first use and replayed on later
  `MetalContext` constructions. Cold startup pays once; hot startup reuses
  the manifest.
- **Numerics report + recovery policy.** Each training step produces a
  `NumericsReport` summarising activation/gradient health. The runtime can
  skip, roll back, or fall back to a safe-family kernel when anomalies are
  detected.
- **Frozen optimizer-state checkpoint contract.** `OptimizerStateBlob` is
  the binary transport for optimizer state and is versioned. See
  [`docs/CHECKPOINT-CONTRACT.md`](CHECKPOINT-CONTRACT.md).
- **Error-handling contract.** The public non-throwing path returns
  `tmnn::Result<T> = std::expected<T, DiagnosticInfo>`; the throwing wrappers
  remain available for callers who prefer exceptions. See
  [`docs/ERROR-HANDLING.md`](ERROR-HANDLING.md).

---

## 5. Platform notes

- macOS Apple Silicon is the primary supported platform. Metal device,
  Foundation framework, and the MSL kernel pipeline are all required.
- Linux and other non-Apple builds compile against a Metal device stub
  (`metal_device_stub.cpp`). The library configures and builds, but
  GPU-runtime tests are skipped.
- The default runtime currently requires `spatial_dims == 3 || 4`. The
  flagship sample lifts 2-D image coordinates to `(x, y, 0)` to satisfy this.
- C++23 is required (`std::expected` is part of the public error contract).

---

## 6. Python binding

The optional `tiny_metal_nn` Python package wraps the C++ `Trainer` via
pybind11. Source: `src/python/tmnn_pybind.cpp`. Build via
`pip install -e ".[dev]" --no-build-isolation` (scikit-build-core invokes
the same CMake project the C++ build uses, with
`-DTMNN_BUILD_PYTHON_MODULE=ON`).

### 6.1 Surface

```python
import tiny_metal_nn as tmnn

with tmnn.Trainer.from_config(config_dict, n_input=3, n_output=1) as t:
    loss = t.training_step(input_array, target_array)   # numpy or torch CPU
    output = t.inference(input_array)                    # owned numpy array
```

Public symbols (`tiny_metal_nn/__init__.py`):

| Name | What |
|---|---|
| `Trainer` | the trainer class — `from_config`, `training_step`, `inference`, `close`, `__enter__` / `__exit__`, `step`, `is_gpu_available`, `closed`, `__repr__`, `summary` |
| `ClosedError` | raised when `training_step` / `inference` is called on a closed trainer |
| `ConcurrentTrainingStepError` | raised when two threads call `training_step` (or one trains while another calls `inference`) on the same trainer |
| `ConfigError` | subclass of `ValueError`; raised by `from_config` for schema violations |
| `DTypeError` | subclass of `TypeError`; raised when a non-`float32` array is passed to `training_step` / `inference` |
| `__version__` | `"0.1.0.dev0"` (matches the CMake project version) |

### 6.2 Wrapper invariants

The Python `Trainer` is a `PyTrainer` C++ wrapper that adds two things on
top of the C++ `tmnn::Trainer`:

- **`closed` flag.** `close()` synchronizes pending GPU work and resets the
  inner `unique_ptr`; subsequent calls raise `ClosedError`. Idempotent.
  The C++ destructor still runs as a fallback if the user forgets to
  close (and the destructor itself drains pending work — `~Trainer()` in
  `include/tiny-metal-nn/trainer.h`).
- **Atomic `in_flight` flag.** Single-shot guard against concurrent
  `training_step` / `inference` calls. The flag uses
  `compare_exchange_strong` so concurrent callers see an explicit
  `ConcurrentTrainingStepError`, not silent serialization.

### 6.3 GIL release boundary

`training_step` and `inference` parse arguments under the Python GIL
(numpy / torch tensor introspection requires it), then explicitly release
the GIL across the GPU dispatch via a manual `py::gil_scoped_release`
scope. This shape — manual release rather than `py::call_guard` — is
what keeps the argument parsing safe and the GPU work non-blocking for
other Python threads.

### 6.4 Tensor input handling

Both `training_step` and `inference` accept any object that supports the
numpy `__array__` protocol — numpy arrays directly, and torch CPU
tensors zero-copy. Non-CPU tensors (MPS, CUDA) are rejected with an
explicit message pointing at `.cpu()`; silent staging is not done. dtype
must be `float32`; non-contiguous storage is rejected with a `.contiguous()`
hint. See `coerce_to_numpy` and `check_float_carray` in
`src/python/tmnn_pybind.cpp`.

### 6.5 Error message format

Errors raised from the binding follow a three-segment structure:

```
ConfigError: <head — the original C++ diagnostic>
  In:     <JSON path or argument name where the error originated>
  Common: <suggested correction or example of valid input>
  See:    docs/TCNN-MIGRATION-GUIDE.md § 10
```

The `Common` lookup table for config paths lives at
`kConfigPathCommon` in `src/python/tmnn_pybind.cpp`. Add an entry there
when introducing a new schema field.

---

## 7. Migration tooling

A separate three-piece set helps users move from `tinycudann` (CUDA) to
`tiny_metal_nn` (Apple Metal). All three pieces consume the same rule
tables in `tools/migrate_rules.py`, which mirror the JSON schema diff
documented in the configuration freeze.

| Piece | What it does |
|---|---|
| `tools/migrate_rules.py` | translation rule tables (`IDENTICAL_FIELDS`, `ALIAS_RULES`, `REJECT_RULES`) plus `translate_config(tcnn_dict) → (tmnn_dict, diagnostics)`. Pure Python, no GPU. |
| `tools/migrate_tcnn.py` | CLI: `tmnn-migrate <file.py> [--check / --diff / --output ...]`. Uses libcst for source rewriting (preserves comments + blank lines). Mechanically rewrites `import tinycudann` and the canonical 5-line training loop body; flags `tcnn.NetworkWithInputEncoding(...)` calls for human / AI follow-up. |
| `.claude/skills/tcnn-to-tmnn.md` | Claude Code skill that consumes the same rules and handles the AST-uncovered cases (custom losses, multi-file projects, judgement calls). |

A worked example pair lives at `examples/migrated/sphere_sdf/`:

- `tcnn/train.py` — original tinycudann (CUDA) version, runnable on a
  CUDA box.
- `tmnn/train.py` — migrated tiny-metal-nn (Apple Metal) version,
  runnable on Apple Silicon.
- `README.md` — step-by-step record of how the migration was performed.

`tests/python/test_examples_convergence.py` enforces that
`tmnn/train.py` reaches `final_loss < 0.01` after 50 steps so accidental
regressions to the binding / runtime are caught at PR time.

---

## 8. Build + dev workflow

### 8.1 C++ only

```bash
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg

cmake -S . -B build -DTMNN_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build -V
```

### 8.2 With the Python binding

```bash
python3.13 -m venv .venv
.venv/bin/pip install scikit-build-core pybind11
VCPKG_ROOT=~/vcpkg .venv/bin/pip install -e ".[dev]" --no-build-isolation

.venv/bin/python -m pytest tests/python/ -v
```

The editable install reuses `build/cp313-cp313-macosx_15_0_arm64/` as a
persistent CMake binary directory; subsequent edits to `src/python/`
auto-rebuild on the next `import tiny_metal_nn`.

### 8.3 Slow-test labeling

Stress / convergence / oracle tests are tagged `slow;deep` in
`tests/LabelSlowTests.cmake`. CI can run them on a separate cadence via
`ctest -L slow` (or exclude them via `-LE slow`).
