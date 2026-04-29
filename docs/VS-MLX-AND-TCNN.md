# tiny-metal-nn vs MLX and tiny-cuda-nn

> Scope: outward-facing product/runtime comparison on the axes this repo
> actually claims today  
> Related: [`docs/BENCHMARKS.md`](BENCHMARKS.md),
> [`docs/TCNN-MIGRATION-GUIDE.md`](TCNN-MIGRATION-GUIDE.md)

---

## 1. What this document is and is not

This document answers the practical question:

> Why use `tiny-metal-nn` on Apple Metal instead of defaulting to `MLX`, and
> how should it be positioned relative to `tiny-cuda-nn`?

It is intentionally **narrower** than a full cross-project bake-off:

- it compares the systems on the product/runtime axes this repo explicitly
  claims
- it cites only benchmark evidence that this repo has actually published
- it does not include a reproducible same-machine `MLX` baseline yet
- it does **not** pretend that the repo has published a kernel-equivalent,
  matched-hardware `tiny-cuda-nn` row on Apple hardware

Closing the remaining gap is on the roadmap; see
[`STATUS.md`](../STATUS.md).

---

## 2. Short answer

- **`MLX`** is the natural default when you want a broad Apple-native tensor
  framework and the freedom to assemble arbitrary operator graphs. It is more
  mature and broader in scope than `tiny-metal-nn`.
- **`tiny-cuda-nn`** is the natural choice on NVIDIA / CUDA. It is the more
  established fused-small-network runtime in that ecosystem.
- **`tiny-metal-nn`** may fit on Apple Metal when you specifically want a
  small-network runtime with:
  - explicit family selection (`FullyFusedMetal`, `TiledMetal`,
    `SafeDebugMetal`)
  - inspectable planner reasons
  - manifest-backed prewarm / hot-path reuse
  - numerics telemetry and a bad-step recovery policy
  - a frozen optimizer checkpoint contract

`tiny-metal-nn` is **not** trying to replace a general tensor framework. Its
scope is narrower: a small-network runtime on Apple Metal, for the workloads
this repository actually ships and measures.

---

## 3. Comparison table

The table below stays on the axes that are explicitly implemented and documented
in this repo.

| Axis | `tiny-metal-nn` | `MLX` | `tiny-cuda-nn` |
|------|------------------|-------|----------------|
| Primary role | Small-network runtime/product surface on Apple Metal | General Apple-native tensor / model framework | Small-network runtime/product surface on CUDA |
| Primary hardware story | Apple GPU families through Metal | Apple hardware through MLX runtime | NVIDIA GPUs through CUDA |
| Config onboarding in this repo | Typed builders plus `tiny-cuda-nn`-style JSON canonicalization in `factory_json.h` | Not the onboarding surface this repo targets | Baseline source vocabulary for the JSON/config portability story |
| Execution-family selection | Explicit `NetworkPlan` with `selected_family`, candidate families, and fallback reasons | No tmnn-style planner contract is part of this repo's comparison story | Specialization/runtime choice exists, but not surfaced here as a `NetworkPlan`-style product contract |
| Hot-path reproducibility | `MetalContext` manifest snapshot/load/prewarm contract | Not a repo-published product contract on this axis | Not a repo-published manifest/prewarm contract on this axis |
| Numerics reporting and recovery | Public numerics report, rollback / skip / safe-family retry policy | Outside this repo's comparison surface | Outside this repo's comparison surface |
| Checkpoint contract | Frozen optimizer-state transport + compat JSON envelope | Outside this repo's comparison surface | Outside this repo's comparison surface |
| Extension / product story | Public extension SDK + planner/runtime contract | Framework-style extensibility, but not the product surface this repo is benchmarking against | Runtime-oriented extension story on CUDA |
| Published benchmark evidence in this repo | Standalone benchmark binary at `tests/benchmarks/tmnn_runtime_benchmarks.cpp`; pre-measured comparison numbers intentionally not published yet | Not yet measured under this repo's harness | Not yet measured under this repo's harness |

A reasonable way to read the table:

- `MLX` is broader than `tiny-metal-nn` and is the right tool for general
  Apple-native tensor work.
- `tiny-cuda-nn` is the closest conceptual peer on a different platform.
- `tiny-metal-nn` is narrower than both. The axes it tries to make explicit
  are operational (planner / manifest / numerics / checkpoint contract), not
  raw-throughput claims, and not "the same kind of thing as MLX".

---

## 4. Why not MLX?

`MLX` is the reasonable default if your problem is:

- "I need a broad tensor framework on Apple hardware"
- "I want to build arbitrary models out of general operators"
- "I value framework flexibility more than a fixed small-network runtime
  contract"

`tiny-metal-nn` may be a better fit when your problem looks more like:

- "I have a small-to-medium fused neural workload on Apple Metal"
- "I want family selection and fallback diagnostics surfaced as part of the
  API"
- "I want hot-path startup and runtime behavior to be reproducible through
  manifest reuse instead of hidden heuristics"
- "I want explicit numerics telemetry / recovery semantics in the training
  runtime"

A short answer to "why not MLX?":

> `tiny-metal-nn` is not trying to be a general framework. It is trying to be
> a small-network runtime on Apple Metal that exposes its planner, manifest,
> numerics, and checkpoint surfaces as part of the contract.

That trade matters most when predictable startup, explicit family selection,
and manifest reuse are values you care about. If they are not, MLX is likely
the better tool.

---

## 5. Why compare against tiny-cuda-nn at all?

`tiny-cuda-nn` remains the clearest reference point for the abstraction class
this repo is targeting:

- encoding + network composition
- small-network fused execution
- config portability for the common onboarding shapes
- a runtime-centric product story rather than a general tensor graph

The honest positioning is:

- on **CUDA / NVIDIA**, `tiny-cuda-nn` remains the natural baseline
- on **Apple Metal**, `tiny-metal-nn` tries to bring that class of product
  surface to the native Apple GPU stack
- tmnn only claims to go beyond `tiny-cuda-nn` on the axes it has actually
  productized here:
  - planner transparency
  - manifest/prewarm reuse
  - numerics recovery observability
  - explicit checkpoint and extension contracts

This is an Apple-Metal claim, not a universal "tmnn beats tcnn everywhere"
claim.

---

## 6. What tiny-metal-nn has actually published today

Current public evidence lives in:

- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — methodology and how to run the
  benchmark binary
- `tests/benchmarks/tmnn_runtime_benchmarks.cpp` — the standalone benchmark
  binary; build and run it to get GPU-measured medians on your Apple Silicon
  device

We are intentionally **not publishing pre-measured "reference numbers"** in
this repository. Earlier internal measurements existed but were collected
under a test harness that is not part of the public source tree, and the
specific workload labels and exact values from those runs are not currently
reproducible from what is committed here. Restoring a publishable, fully
reproducible board is on the roadmap.

---

## 7. Same-machine comparison against MLX and tcnn

We do not yet publish a same-machine, same-task comparison slice in this
repository. An earlier internal slice against an MLX dense ReLU MLP baseline
existed but the measurement script is not in the public tree, so we cannot
honestly claim reproducibility from this repo today. A reproducible
same-hardware comparison against MLX (architecture-equivalent model on both
sides) is on the roadmap; a `tiny-cuda-nn` row on Apple is structurally
impossible because `tiny-cuda-nn` is CUDA-only.

In the absence of a published comparison, the positioning argument has to
stand on the product surface — explicit planner/family selection,
manifest-backed prewarm, numerics report, frozen optimizer-state checkpoint
contract — not on raw-throughput claims. See §1 and §3 for the axes we do
defend today.

---

## 8. Current recommendation

Use `tiny-metal-nn` when all of the following are true:

- you are shipping on Apple hardware
- your workload fits the repo's small-network/runtime story
- you care about startup/hot-cache behavior as a product contract
- you want planner/autotune/numerics/checkpoint behavior to be inspectable

Use `MLX` when you want a more general framework.

Use `tiny-cuda-nn` when you are solving the same class of problem on CUDA.

---

## 9. What remains open

This repo still needs the following before the comparison story is fully
honest:

- a stronger, more architecture-matched tmnn-vs-MLX comparison slice
- any CUDA-side `tiny-cuda-nn` performance rows still require a separate NVIDIA
  environment; the same-machine Apple row will remain unavailable by design
- a broader cross-device Apple validation matrix (`M1` / `M2` / `M3` families
  where available)
- more than one external workload using only installed/public package surfaces

Until then, this document should be read as the **current product-positioning
note**, not as the final external bake-off verdict.
