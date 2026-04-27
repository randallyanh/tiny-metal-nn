# tiny-metal-nn Status (2026-04-27)

A snapshot of what is and is not in the box today. If a claim elsewhere in this repository conflicts with what is recorded here, this document is the one we update first; the others may lag. Refreshed whenever a readiness item lands or a known limitation is fixed.

## What works today

| Capability | Status |
|---|---|
| Fully-fused MLP families (`FullyFusedMetal`, `TiledMetal`, `SafeDebugMetal`) | ✅ |
| Hash-grid encoder (`HashGrid`) and rotated multi-resolution variant (`RMHE`) | ✅ |
| `factory_json::canonicalize_model_config` — tcnn-compatible JSON | ✅ |
| Network planner (`NetworkPlan` with explicit family selection + fallback reasons) | ✅ |
| `MetalContext` runtime — Metal device, pipeline registry, batch pool | ✅ |
| Manifest-backed kernel prewarm (record kernels at first use, replay on later boots) | ✅ |
| Frozen optimizer-state checkpoint contract — see [`docs/CHECKPOINT-CONTRACT.md`](docs/CHECKPOINT-CONTRACT.md) | ✅ |
| Numerics report + bad-step recovery policy | ✅ |
| Built-in extension adapters: DNL, RMHE, 4D, standard SDF, multi-output MLP | ✅ |
| Fused Adam / AdamW kernels, fp16 + fp32 paths | ✅ |
| `samples/mlp_learning_an_image.cpp` — flagship sample, 319 LOC, matches tcnn convention | ✅ |
| Standalone GPU-measured benchmark binary (`tests/benchmarks/tmnn_runtime_benchmarks.cpp`) producing real-Metal medians on M1 Pro | ✅ |
| gtest-driven test suite (~10K LOC of tests), runs under `ctest` | ✅ |
| Apache-2.0 license + contributor docs (CONTRIBUTING / SECURITY / CODE_OF_CONDUCT) | ✅ |

## What does not work yet

| Item | Status |
|---|---|
| Comparison benchmark vs MLX on the same hardware (architecture-equivalent models on both sides) | Not yet measured / published |
| Comparison benchmark vs tcnn on equivalent hardware | Not yet measured |
| `examples/nerf_synthetic/` flagship sample | Not yet implemented |
| `examples/hash_grid_sdf_fitting/` flagship sample | Not yet implemented |
| Cross-device benchmark validation (M2 / M3 / M4) | Numbers only on M1 Pro |
| CI matrix (macOS Apple Silicon + Linux Metal-stub) | Not yet wired |
| `MAINTAINERS.md` with SLA | Not yet written |
| Python bindings | Not yet implemented |

## Known limitations

- **macOS Apple Silicon is the primary supported platform.** Linux configures and builds with a Metal stub backend; GPU-runtime tests are skipped on non-Apple hosts. There is no CUDA backend in tmnn (use tiny-cuda-nn directly on NVIDIA).
- **Spatial dims constraint.** The default runtime currently requires `spatial_dims == 3 || 4`. The `mlp_learning_an_image` sample lifts 2D image coordinates to `(x, y, 0)` to satisfy this.
- **One device benchmarked.** All internal performance measurements have been on a single Apple M1 Pro. M2 / M3 / M4 numbers are not published. Cross-device validation is on the roadmap.
- **Pre-1.0 API.** Public APIs may change without deprecation periods until the v0.1.0 tag. Breaking changes happen on `main` without notice during this phase.
- **Sibling consumer (slangcsg) is currently private.** [slangcsg](https://github.com/randallyanh/slangcsg) consumes `tmnn_core` + `tmnn_runtime` for differentiable CSG / SDF compute and is being prepared for its own open-source release on its own timeline. The contract on tmnn's side is captured in [`docs/CHECKPOINT-CONTRACT.md`](docs/CHECKPOINT-CONTRACT.md) and [`docs/EXTENSIBILITY-DESIGN.md`](docs/EXTENSIBILITY-DESIGN.md).

## Performance numbers

We are intentionally **not publishing pre-measured "reference numbers" in
this repository** until a same-hardware comparison story against MLX
(architecture-equivalent) and `tiny-cuda-nn` (cross-platform) lands. The
current public surface is the standalone benchmark binary at
`tests/benchmarks/tmnn_runtime_benchmarks.cpp` — build and run it on your
Apple Silicon device to get GPU-measured medians. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the methodology and
[`docs/VS-MLX-AND-TCNN.md`](docs/VS-MLX-AND-TCNN.md) for the positioning we
believe we can defend today.

## What tmnn is not

This list is here to save reader time:

- Not a general ML framework (use MLX, PyTorch, or JAX)
- Not a generic GPU compute substrate (rasterization, CSG, meshing, image processing all out of scope; consume `tmnn_core` + `tmnn_runtime` from a separate library if you need those)
- Not an execution engine that takes arbitrary kernels and dispatches them — it owns its own kernels
- Not a drop-in replacement for tinycudann's Python API. tmnn is C++; tcnn-compatible at the JSON-config level, not at the PyTorch-binding level.
- Not benchmarked against MLX or tcnn yet on the same hardware with architecture-equivalent models, so any "X% faster than Y" claim is unsupported until that comparison is published.

## Realistic outcome assessment

tmnn's most identifiable user is "researcher porting a tcnn-based codebase to Apple Silicon." This is a real audience but a small one — most are likely to try MLX first. Comparable open-source projects with similar audience scope and Apple-only targeting typically reach 200–500 GitHub stars and are abandoned within 24 months. We are recording this as the realistic baseline outcome, not the worst case. Anything beyond this is upside, not entitlement. The same-hardware architecture-equivalent comparison against MLX is the work that determines whether tmnn has a defensible niche or is dominated by MLX in practice.

## Update procedure

Bump the date at the top of this file every time a readiness item lands. Move items between "What works today" and "What does not work yet" as work completes. If you are unsure whether a claim belongs in the README or in this document, the answer is: this document.
