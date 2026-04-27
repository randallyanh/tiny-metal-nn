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
