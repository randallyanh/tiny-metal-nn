# tiny-metal-nn Operator & Extensibility Design

> Date: 2026-03-16
> Status: output seam / probe mode / declarative loss config landed; the inference-only narrow API stays demand-driven.

---

## 0. Fusion as the design center

**tmnn's design center is the `MLPKernelEmitter`-driven fused kernel; it is not an operator framework.**

This is the first principle of the document. Every extension design tries to satisfy:

- the production path (default kernel) takes zero extra cost
- extensions go through the adapter seam, not through a public operator seam
- observability, fallback, and instrumentation are explicit opt-ins, kept out of the default path

### Performance evidence

Profiling on the standard 3D SDF + hash-grid + MLP scenario:

- fused kernel: 660us GPU + 200us dispatch = **860us**
- if split into 5 separate operators: 660us GPU + 5×200us dispatch + 5×100us memory = **2,160us (2.5×)**

A more important data point: the default runtime once dropped from 130ms back down to 2.3ms. The root cause was not operator math; it was **debug/recovery semantics leaking into the default fast path**. This proves any non-production logic that lands in the default path produces an order-of-magnitude regression.

> Note: the numbers above are illustrative of the design direction, not a public benchmark. To measure on your own machine, run `./build/tests/tmnn_runtime_benchmarks` (see [`docs/BENCHMARKS.md`](BENCHMARKS.md)).

---

## 1. Extension model: adapter-centric

### The existing skeleton

tmnn already has the right extension skeleton:

```
TrainingAdapter (virtual interface)
  ├─ schema()                 → declares parameter layout + buffer geometry
  ├─ loss_config()            → declares the built-in loss family (L2 / L1 / Huber / Cosine)
  ├─ configure_compile_spec() → patches the baseline compile policy
  ├─ pack_batch()             → controls input packing
  ├─ fill_train_params()      → controls training-parameter layout
  ├─ adam_config()            → controls optimizer parameters
  ├─ result_metrics()         → controls output metrics
  └─ pack_config_tail()       → controls extra config tail
```

**Loss is part of training semantics.** It belongs with batch packing, target structure, and extra-parameter layout — and that is exactly what the adapter is for.

### Wrong directions (abandoned)

| Direction | Problem |
|-----------|---------|
| `LossSnippet` (MSL fragment injection) | Parallel to `TrainingAdapter`; two extension systems will fight |
| `ComputeGraph` graph-compilation framework | 6+ person-months; a general compiler loses to a focused emitter |
| Fine-grained operators (matmul, relu) | Splitting them up means losing the fusion advantage = a slow PyTorch |
| A public runtime training-operator API | Turns tmnn into an operator-graph framework; drifts away from the fusion doctrine |

### Correct direction A: simple loss lives on the adapter seam (landed)

```cpp
enum class LossKind { L2, L1, Huber, Cosine };

struct LossConfig {
  LossKind kind = LossKind::L2;
  float huber_delta = 1.0f;
};

class TrainingAdapter {
public:
  // ... existing methods ...

  /// Declare the loss family; the runtime lowers it into the emitted kernel
  /// at compile time.
  virtual LossConfig loss_config() const { return {}; }
};
```

The result:

- the default path is unchanged (L2 loss is inlined; zero overhead)
- simple custom losses are declared via the adapter (L2 / L1 / Huber, plus Cosine on the generic multi-output path)
- the runtime lowers loss ownership into the emitted kernel during pipeline construction
- no second extension system is introduced

The current code-level boundary is honest:

- `TrainingAdapter::loss_config()` has landed; the default runtime validates the adapter schema and compile preferences at construction time, then lowers the declarative loss into `TrainerConfig` plus a matching built-in `Loss` object.
- This seam in `DefaultRuntime` is still a **narrow seam**: it only validates and lowers at construction. The full lifecycle integration of `pack_batch()` / `fill_train_params()` / `adam_config()` / `result_metrics()` / `pack_config_tail()` still belongs to adapter-native runtimes.
- To keep the public surface honest, `create_loss(...)` / `create_from_config(...)` / `create_trainer(model, loss, optimizer, ...)` now also accept the built-in `L2` / `L1` / `Huber` directly.

If a future requirement genuinely calls for "a real custom formula", the answer is still not a bare `LossSnippet`. That can only come **as a follow-up extension** and only after a fixed ABI is defined (for example: agreed `prediction` / `target` / `loss_out` / `d_loss_d_prediction_out` names plus validation rules), with no dependency on the emitter's internal local-variable names. **Without an ABI, MSL fragment injection is not the v1.0 plan.**

### Correct direction B: non-local objectives go through the output seam

`neulat`-style workloads do not fit single-kernel fusion (stress estimation requires inter-point spatial relationships). The currently practical path is **outer-loop two-stage optimization**, not in-kernel loss replacement:

```
Stage 1: tmnn standard training        → fits SDF + density field (multi-output adapter)
Stage 2: Trainer::evaluate() / evaluate_with_gradient()
                                       → exports intermediate values
Stage 3: external TPMS / FEM / compliance computation
                                       → produces new supervision / modified target
Stage 4: back to Trainer::training_step()
```

There are **two real integration paths today**:

- the looser-coupled prototype: `evaluate()` / `evaluate_with_gradient()` exports intermediate values and accepts the extra device↔host round trip
- the tighter training loop: use the already-landed external-gradient output seam directly

The landed training seam is **not** a public module-level operator API — it cuts only at the network output:

```cpp
auto pass = trainer.forward_for_training(input, target, N);
// external TPMS / FEM / compliance -> dL/dy
trainer.backward_from_output(pass, d_output);
trainer.optimizer_step();
```

The point here is not "split the training kernel"; it is to introduce an **output seam plus an opaque pass**. Internally, encoding + MLP backward stay fused; only the output-layer gradient injection point is exposed.

The current boundaries, stated honestly:

- on the split path, the loss semantics are determined by the external `d_output`; `TrainerConfig.loss_kind` mainly affects the fused / internal-loss kernels.
- richer multi-output / domain-specific output contracts still need an adapter-aware caller. That said, the generic multi-output adapter-backed split path now has a `target_dims = 32` external-gradient convergence witness, so it is no longer just a single-output SDF-style proof.
- the public/runtime built-in loss surface now supports `L2` / `L1` / `Huber`, plus `Cosine` on the generic multi-output path.
- `Cosine` support is bounded today: it requires the generic multi-output path with `target_dims >= 2`, `schema.reduction_terms == 1`, and it does not support DNL / `bc_dim_count` decomposition.
- when a host round trip is acceptable, `evaluate()` / `evaluate_with_gradient()` remain the simpler prototyping route.

### Output seam: the minimum landed contract

This API set is **a real capability today**. It is backed by five implementation lines:

1. **External-gradient kernel path is landed.**
   - `ForwardForTraining` / `BackwardFromExternalGrad` are independent kernel roles.
   - Forward still flows through the existing encoding + MLP path; backward reads the output gradient from an external `d_output` buffer instead of recomputing the loss inside the split path.
   - Dispatch geometry is locked by `KernelDispatchContract` plus the plan verifier, not by ad-hoc launch parameter assembly.

2. **Opaque-pass contract is landed.**
   - `forward_for_training()` returns a `ForwardPass`, not a raw output tensor.
   - `ForwardPass` exposes `valid()` / `output_count()` / `output_dims()` / `output(i)` / `output_data_ptr()`, threading together the context this backward needs.
   - The default runtime keeps a simple constraint: each trainer holds at most one in-flight split-path state at a time, rather than turning the runtime into a complex scheduler upfront.

3. **The runtime lifecycle is split into three stages.**
   - `forward_for_training()`: prepare the step lane, run the forward dispatch, produce the output and pass.
   - `backward_from_output()`: upload the external `d_output`, run the backward kernel, produce gradients usable by Adam.
   - `optimizer_step()`: run the Adam apply, sync live/config weights, advance the step counter.
   - The public API and the internal state machine are now consistent — no more "three steps on the surface, one big step underneath".

4. **Numerics / recovery / optimizer state are wired into this path.**
   - The split path and the fused path share the runtime-owned optimizer state and step semantics.
   - `reset_optimizer()` / `export_optimizer_state()` / `import_optimizer_state()` are no longer correct only on the fused path.
   - This path also picked up a split probe readback (`read_last_split_probe()`), keeping the fused/split observability surface symmetric.

5. **Witnesses cover the currently usable boundary.**
   - Correctness witness: when the external `d_output` is derived from the built-in L2 loss, the split path matches `training_step()`.
   - Behavior witness: loss decreases, and the step counter, optimizer state, and `sync_weights` behavior do not drift.
   - External-loss witness: an `L1` / `Huber` gradient supplied by the caller also converges, proving the split path's semantics really are defined by the external gradient.
   - Generic multi-output witness: the adapter-backed split path now has a `32`-output external-L2 gradient convergence proof, covering both flat `d_output` reads and the `schema.reduction_terms` geometry contract.
   - Performance witness: the split-vs-fused cost still has to be measured per scenario; the split path is not a "free feature".

### What it means to "have" this capability

Exposing three public methods is not the same as having the capability. tmnn only counts as having a real external-gradient output seam when all of the following hold:

- `forward_for_training()` returns **a consumable output and a valid pass**.
- `backward_from_output()` consumes the external gradient and produces the right parameter gradients.
- `optimizer_step()` performs a legal update based on that backward's gradients.
- The split path's numerics / optimizer state / step semantics match the existing `training_step()`.
- At least one witness shows that `training_step()` and the external-gradient path behave the same under an equivalent loss.

In short, **"have"** means it is a verifiable, repeatable training capability that can be integrated into an external optimization pipeline like neulat — not just an API surface.

### Implementation decision: the split lives in the tmnn detail/runtime layer

This open question has now converged: the implementation is split along tmnn's own detail/runtime lifecycle, not by duplicating training-step logic in the default/compat layer.

- `training_step_execution` / the dispatch contract / the plan verifier own the binding and geometry contract for the forward and backward kernels.
- `DefaultRuntime` owns the public split API, the step-lane state, and the optimizer-step orchestration.
- The fused path and the split path therefore share recovery / optimizer-state / probe / sync semantics rather than reimplementing them inside a compat layer.

This is the same point our boundary cleanup keeps coming back to: product / compat code should consume the seams tmnn already provides, not re-state the runtime lifecycle.

---

## 2. Opt-in probe mode

### Principle

Observability is **explicitly opt-in**. It does not enter the default path.

This mirrors `NumericsGuard`: `NumericsGuard` already supports `NumericsSamplingMode::Periodic` (check only on sampled steps); probe mode follows the same shape — decided at trainer construction, with the production path completely unaffected.

### Design

The following surface is now landed:

```cpp
struct TrainerConfig {
  bool enable_probes = false;
};

struct ProbeResult {
  uint32_t num_hidden_layers;
  bool has_nan_forward, has_nan_backward;
  float hash_grad_l2;
  float act_max[kMaxLayers];
  float grad_l2[kMaxLayers];
  float output_abs_max, output_min;
};

struct TrainingStepResult {
  std::optional<ProbeResult> probe;
};

std::optional<ProbeResult> Trainer::read_last_split_probe() const;
```

When `enable_probes = true`:

- the compiler emits a training-kernel variant that writes into the probe buffer and forces the scalar training path; `ForwardForTraining` explicitly ignores the probe preference.
- the fused `training_step()` returns the per-step probe via `TrainingStepResult.probe`.
- on the split path, `read_last_split_probe()` returns the probe of the most recent step after `optimizer_step()`.
- the production path (`enable_probes = false`) is untouched — the default fast path emits no probe code.

A complementary note: there is also a **lighter-weight runtime inspection surface** in the current code:

```cpp
std::optional<TrainerRuntimeInspection> Trainer::inspect_runtime() const;
```

It exposes only the **requested-vs-realized specialization and dispatch geometry of the compiled kernel** (entry point, requested vs realized SIMD, TG-cache, `tg_size`, `pts_per_tg`, etc.) — useful for understanding and debugging the specialization contract. It is **not** the per-step numeric probe mode discussed here, and it does not introduce any extra training-time probe buffer or kernel instrumentation.

### Why the implementation is larger than the surface suggests

Probe mode has landed, but it really is not "add a bool plus a read API". It cuts across at least the following five implementation lines:

1. **Emitter / kernel ABI**
   - Add a probe-buffer binding slot to the training kernel.
   - Define which per-layer statistics get written (activation min/max, gradient norm, NaN/Inf flags).
   - Handle in-threadgroup reduction and across-threadgroup writeback so the probe writes don't become the new bottleneck.
   - Guarantee that no probe code is emitted at all when `enable_probes = false`.

2. **Runtime buffer layout / storage policy**
   - Define a new buffer role / offset / byte size / lifetime for the probe data.
   - Decide whether the probe buffer is `CpuVisible` or `GpuOnly + staging`.
   - Wire it into the existing planner / parameter store / binding plan, not as a side channel.

3. **Training-step lifecycle and readback**
   - In `training_step`, weave probe zeroing, writeback, readback, and synchronization timing into both the enqueue and finalize paths.
   - Define that the fused path uses `TrainingStepResult.probe` and the split path uses `read_last_split_probe()`.
   - Handle the boundary cases around async stepping, pending command buffers, `reset_optimizer`, and `sync_weights`.

4. **Public API and semantic contract**
   - `TrainerConfig::enable_probes` is just the entry point; the stable shape of `ProbeResult` still has to be defined.
   - Be explicit about which family / kernel variants support probes and which configurations must reject or downgrade.
   - Define how probes interact with `NumericsGuard`, bad-step recovery, and safe-family retry when enabled together.

5. **Verification and performance regression**
   - Unit tests: probe layout, binding count, readback format.
   - Integration tests: coexistence with recovery / retry / async paths.
   - Performance tests: prove that `enable_probes = false` is zero-impact and that `enable_probes = true` stays within the budgeted overhead.

In other words, probe mode is a **complete tranche**, not a single switch. The right way to implement it is to extend the existing emitter → runtime planner → training-step lifecycle path, while leaving the default fast path untouched.

### Relationship with `NumericsGuard`

| | `NumericsGuard` | Probe Mode |
|---|---|---|
| Purpose | Detect NaN/Inf → trigger recovery | Emit full per-layer stats → debugging / tuning |
| Default | Periodic sampling | Off |
| Performance impact | ~1% (read-only reduction buffer) | 0 when off; depends on probe granularity when on (budgeted at 10–20%) |
| Lives where | Inside `finalize_training_step` | Inside the kernel |

Probe mode complements `NumericsGuard`; it does not replace it. Both can be enabled at the same time.

---

## 3. Inference-only narrow API

### Principle

Open this only when there is a clearly stated user need. Gate it explicitly via `RuntimeAuthority`. Do not turn tmnn into a public operator-graph framework.

### Current state

`Trainer::evaluate()` and `Trainer::inference()` already cover ~99% of inference needs.

### Possible future extension (not in v1.0)

```cpp
// Available only when a RuntimeAuthority is held (gating).
namespace tmnn::inference {

// Stepwise inference — power users only.
struct InferenceContext {
  static InferenceContext create(const RuntimeAuthority &authority);
  void hash_encode(const float *positions, float *features, int N);
  void mlp_forward(const float *features, float *output, int N);
};

}
```

**When to ship it**: when a user asks specifically for stepwise inference (intermediate-feature visualization, feature-level caching, multi-head inference, etc.). Without that demand, do not implement it.

---

## 4. Things we explicitly do not do

| Direction | Reason for dropping | Decision date |
|-----------|---------------------|---------------|
| `ComputeGraph` graph-compilation framework | 6+ person-months; a general compiler loses to a focused emitter | 2026-03-16 |
| Fine-grained runtime operators | Splitting them up = losing the fusion advantage | 2026-03-16 |
| `LossSnippet` as an independent extension system | Parallel to `TrainingAdapter`; the two would fight | 2026-03-16 |
| A public module-level training forward/backward API | Would turn tmnn into an operator graph; threadgroup-local optimizations also cannot be exposed as a stable ABI | 2026-03-16 |
| A public training-runtime operator API | Turns tmnn into an operator framework; drifts off the fusion doctrine | 2026-03-16 |
| External-facing eager-execution semantics | tmnn is compile-time fusion, not runtime scheduling | 2026-03-16 |

---

## 5. Implementation status

```
External-gradient output seam — landed
    ├─ output seam + opaque pass implemented
    ├─ `forward_for_training()` / `backward_from_output()` / `optimizer_step()`
    ├─ external-gradient backward kernel / compiler path landed
    ├─ outstanding-pass state machine landed under the single-trainer simplification
    ├─ numerics / recovery / optimizer-state semantics wired in
    ├─ correctness / convergence / external-loss witnesses landed
    └─ split-vs-fused cost still has to be evaluated per perf scenario

Opt-in probe mode — landed
    ├─ `TrainerConfig::enable_probes`
    ├─ conditional probe instrumentation + runtime readback
    ├─ fused `TrainingStepResult.probe`
    ├─ split `read_last_split_probe()`
    ├─ coexistence tests with recovery / retry / async paths
    └─ default `enable_probes=false` keeps the fast path clean

Adapter-owned declarative loss config — landed
    ├─ `TrainingAdapter::loss_config()`
    ├─ runtime lowers it into the emitted kernel / `TrainerConfig`
    ├─ built-in `L2` / `L1` / `Huber`
    ├─ public `create_loss(...)` / `create_from_config(...)` are in sync
    └─ default runtime stays a narrow seam, not a full adapter-native runtime

Inference-only narrow API — demand-driven
    └─ ship when a user actually asks for it
```

---

## 6. Core principles (final)

1. **Fusion is the design center.** Every design choice serves the fused-kernel path first.
2. **Adapter-centric extensibility.** Extensions go through the `TrainingAdapter` seam; we do not open new public seams.
3. **Opt-in only.** Observability, probes, and instrumentation never enter the default path.
4. **No parallel extension systems.** One extension mechanism (the adapter), not two (adapter + snippet).
5. **Inference may split; training internals stay fused.** When complex external losses are required, only consider an output seam plus an opaque pass; never expose module-level operators.
6. **Ship when asked.** The inference API does not get implemented before someone needs it.
