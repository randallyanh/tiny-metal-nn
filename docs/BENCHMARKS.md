# tiny-metal-nn Benchmark Methodology

> Date: 2026-03-14  
> Primary measurement binary: `tests/benchmarks/tmnn_runtime_benchmarks.cpp`

---

## 1. Purpose

This document defines the **public benchmark contract** for `tiny-metal-nn`.

It exists so that benchmark claims are:

- hardware-qualified
- tied to a named workload
- explicit about cold vs hot behavior
- explicit about autotune / manifest reuse
- reproducible from the checked-in test suite

This is intentionally narrower than a full external bake-off. Cross-device and
outside-the-team validation are tracked in [`STATUS.md`](../STATUS.md) under
"What does not work yet".

---

## 2. Benchmark binary

The public benchmark binary lives at:

- `tests/benchmarks/tmnn_runtime_benchmarks.cpp` (built as a standalone
  executable, not added to `ctest`)

It currently covers:

- small MLP training routes on representative tmnn families
- planner / autotune-search behavior
- forced-family inference throughput
- `default-trainer hot-step` measurements with optional per-step profiling

Build and run it from the build directory with:

```bash
cmake --build build --target tmnn_runtime_benchmarks
./build/tests/tmnn_runtime_benchmarks --smoke      # quick check
./build/tests/tmnn_runtime_benchmarks              # full run
```

For correctness regression after changes (this is `ctest`, not the benchmark
binary):

```bash
ctest --test-dir build --output-on-failure
```

---

## 3. Reproducibility contract

### 3.1 Hardware qualification

Every published board entry must record:

- Apple device name
- `gpu_family`
- macOS build
- benchmark command
- route config and seeds

No benchmark claim is considered public-facing unless it is attached to that
metadata.

### 3.2 Cold vs hot definitions

- **Cold** means a fresh route on a `MetalContext` before the route's first
  compile/cache/materialization event.
- **Hot** means the same route after the first cold build/run has already
  populated runtime caches and autotune decision state on the same
  `MetalContext`.

For the training routes, "hot" also means the selected family came back via
manifest-backed reuse (`NetworkPlan::from_autotune_manifest`).

### 3.3 Variance policy

The board records one hardware-qualified run of the checked-in benchmark suite.

Within that run:

- hot training latency is the median of `24` repeated training steps
- hot startup latency is the median of `3` repeated startup runs
- hot inference latency is the median of `12` repeated evaluations

We currently report medians from the in-suite repeated samples rather than
multi-process confidence intervals. Future board entries must keep the same
policy unless the methodology document is updated.

### 3.4 Autotune policy

`C6` intentionally kept live bounded measured search **opt-in**. The benchmark
story therefore distinguishes between two states:

- **cold discovery**: bounded measured search may run when
  `NetworkFactoryOptions::enable_bounded_autotune_search = true`
- **reproducible hot path**: prior measured decisions are reused from the
  `MetalContext` autotune manifest

Public hot-path claims should be interpreted as **manifest/prewarm-backed**
behavior, not as "fresh live search always picks the same family every run."

---

## 4. Route catalog

### 4.1 Small training routes

These routes exercise the real planner/autotune/runtime contract and verify:

- the cold path compiles
- the hot path does not recompile
- manifest-backed reuse is active
- loss improves on a fixed named workload

Common settings:

- batch size: `1024`
- bounded search batch size: `1024`
- bounded search measurement steps: `2`
- hot training samples: `24`

Routes:

| Route label | Model family | Config source | Weight seed | Sample seed |
|-------------|--------------|---------------|-------------|-------------|
| `standard-3d` | standard HashGrid + scalar MLP | `smallNeuralSDFConfig()` | `42` | `7` |
| `four-d` | 4D HashGrid + scalar MLP | `small4DNeuralSDFConfig()` | `42` | `7` |
| `rmhe` | rotated hash grid + scalar MLP | `smallNeuralSDFConfig()` | `42` | `7` |

### 4.2 Time-to-target-loss

Named workload:

- `standard-3d-sphere-small`

Policy:

- cold route runs bounded measured search
- hot route must reuse the measured manifest choice
- target loss is `90%` of the first hot-step loss
- max budget is `48` hot training steps

Settings:

- batch size: `1024`
- weight seed: `42`
- sample seed: `77`

### 4.3 Startup benchmark

Named workload:

- `standard-3d-sphere-small`

Policy:

- cold startup = `build_runtime_bundle(...)` plus the first `training_step(...)`
- hot startup = median of `3` repeated bundle+first-step runs on the already
  warmed `MetalContext`

Settings:

- batch size: `1024`
- weight seed: `91`
- sample seed: `97`

### 4.4 Inference benchmark

Named workload:

- `standard-3d-inference`

Policy:

- forced-family comparison between `FullyFusedMetal` and `TiledMetal`
- cold inference = first `evaluate(...)`
- hot inference = median of `12` repeated evaluations

Settings:

- point count: `16384`
- weight seed: `123`
- position seed: `131`

### 4.5 Parity / cache proof routes

These are part of the public suite because they explain *why* the timing claims
are trustworthy:

- `RepresentativeFamiliesMeetHotCacheCompileMissTarget`
- `FullyFusedAndTiledTrainingStayNumericallyAligned`

They are proof routes, not leader-board rows.

---

## 5. Public autotune and prewarm usage

The public API for manifest-backed reuse is:

- `tmnn::load_autotune_manifest(path)`
- `tmnn::save_autotune_manifest(path, manifest)`
- `MetalContext::prewarm_autotune_manifest(...)`
- `MetalContext::snapshot_autotune_manifest()`

Typical flow:

```cpp
using namespace tmnn;

auto ctx = MetalContext::create();

AutotuneManifest prior = load_autotune_manifest("tmnn-autotune.json");
ctx->prewarm_autotune_manifest(prior);

NetworkFactoryOptions opts;
opts.metal_context = ctx;
opts.enable_bounded_autotune_search = true;
opts.autotune_search_batch_size = 1024;
opts.autotune_search_measure_steps = 2;

auto trainer = create_trainer_from_config(n_input, n_output, config, opts);

save_autotune_manifest("tmnn-autotune.json",
                       ctx->snapshot_autotune_manifest());
```

Interpretation:

- if there is no prior manifest entry, tmnn may run a bounded live search
- once a measured decision exists, later runs can reuse it through the same
  `MetalContext` or a persisted manifest
- benchmark hot-path claims should be read through that reuse contract

---

## 6. Refresh procedure

To refresh the public board:

1. Build the benchmark binary: `cmake --build build --target tmnn_runtime_benchmarks`.
2. Run `./build/tests/tmnn_runtime_benchmarks`.
3. Record the emitted route lines exactly.
4. Record `sw_vers`.
5. Note the measurement context (device, command, OS version) in the PR description.
6. Re-run full `ctest --test-dir build --output-on-failure` (correctness regression) before claiming closure.

---

## 7. Current limitations

- The checked-in board currently contains one named Apple device, not the full
  representative device spread.
- Cross-device expansion is on the roadmap; see [`STATUS.md`](../STATUS.md).
- The board is intended to compare tmnn behavior on the same repo/test contract;
  it is not yet a full tmnn-vs-MLX or tmnn-vs-tiny-cuda-nn comparative study.
