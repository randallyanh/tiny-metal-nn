# Contributing to tiny-metal-nn

Thanks for your interest. This document covers what gets accepted, the workflow, and the conventions the codebase already follows.

## Scope

`tiny-metal-nn` is a Metal-native, tiny-cuda-nn-inspired runtime for hash-grid + MLP neural-field training and inference on Apple GPUs. The library is intentionally narrow; the "out of scope" list below is part of the contract.

In scope:

- Fully fused MLP family kernels (FullyFusedMetal, TiledMetal, SafeDebugMetal) — accuracy, performance, additional families with clear use case
- Hash-grid encoders and rotated multi-resolution variants
- `factory_json::canonicalize_model_config` — additions that increase tcnn JSON-config compatibility
- Network planner improvements (better fallback reasons, more inspectable diagnostics)
- Performance work backed by `tests/benchmarks/tmnn_runtime_benchmarks.cpp` numbers
- Documentation and reproducibility improvements
- Additional flagship samples that follow the `samples/mlp_learning_an_image.cpp` pattern (≤ 400 LOC, runnable from a clean clone, mirrors a tcnn / instant-ngp-style workload where applicable)
- Python binding work that follows the v1.0 design contract in [`docs/know-how/006-python-binding-design.md`](docs/know-how/006-python-binding-design.md) (single `tiny_metal_nn` namespace, fused `Trainer.training_step`) and the migration tooling deliverables (`tools/migrate_tcnn.py` CLI, `.claude/skills/tcnn-to-tmnn.md` skill, `examples/migrated/`)

Out of scope:

- General-purpose autodiff (use JAX, PyTorch, or `dr.jit`)
- General-purpose tensor framework (use [MLX](https://github.com/ml-explore/mlx))
- Generic GPU compute substrate — rasterization, CSG, meshing, image processing all belong to consumer libraries (e.g., [slangcsg](https://github.com/randallyanh/slangcsg))
- CUDA backend (use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) directly on NVIDIA)
- A `tiny_metal_nn.compat.tcnn` shim namespace or `nn.Module` wrapping path — both rejected in `006-python-binding-design.md` v2 §1 / §5.1; tcnn compatibility is JSON-config layer (today) plus the migration tooling three-piece set (v1.0)
- Medical-domain or FDA-regulated code — lives in private downstream repositories

If you are unsure whether something fits, open an issue with the proposal before writing code.

## Development environment

Required:

- macOS 14+ on Apple Silicon (primary target). Linux configures and builds with a Metal stub backend; GPU-runtime tests skip.
- CMake 3.20+
- [Vcpkg](https://github.com/microsoft/vcpkg) — set `VCPKG_ROOT` or place a checkout at `./.deps/vcpkg` (see [`README.md`](README.md) "Build")
- A C++23-capable compiler (Apple Clang on macOS; GCC 12+ or Clang 16+ on Linux)

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build -V
./build/samples/mlp_learning_an_image                      # flagship sample (built-in config)
```

## Code style

Formatting is captured in `.clang-format` (LLVM-derived: 2-space indent, 80-col, K&R braces, `&` attached to the variable name). For new code:

```bash
clang-format -i path/to/file.cpp     # format in place
clang-format -n path/to/file.cpp     # check only
```

The existing code is mostly aligned with `.clang-format` but not perfectly; if a PR touches a file, running `clang-format` on the changed regions is welcome but reformatting unrelated code in the same PR is not.

Conventions on top of formatting:

- C++23. `[[nodiscard]]` on every function whose return value carries information; the codebase treats this as a rule.
- `Result<T>` / `DiagnosticCode` for fallible APIs (see [`docs/ERROR-HANDLING.md`](docs/ERROR-HANDLING.md)). Throwing exceptions across module boundaries is reserved for unrecoverable invariant violations.
- One thing per file when possible; prefer narrow public headers and wide-but-private implementation files.
- File-level Doxygen `@file` + `@brief` on every source file.
- Modern idioms: `std::variant` for tagged sum types, `std::expected` for fallible APIs, RAII for Metal device / queue / buffer lifetimes, `auto` only where the type is obvious from the right-hand side.
- No `using namespace std`. Inside test files `using namespace tmnn` is acceptable.

MSL kernels:

- Generated through `KernelSpec` / `MLPKernelEmitter` whenever possible. Hand-written `.h` kernel snippets exist for cases the emitter cannot express; document the reason in the file's header comment.
- All kernel changes must include a test in `tests/unit/tmnn/` that exercises the affected family.

JSON config:

- New fields go through `factory_json::canonicalize_model_config` and document the corresponding `DiagnosticCode` if validation can fail. Maintain tcnn JSON-config compatibility wherever practical (see [`docs/TCNN-MIGRATION-GUIDE.md`](docs/TCNN-MIGRATION-GUIDE.md)).

Comments:

- Default to writing none. Only add a comment when the *why* is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific bug, or behavior that would surprise a reader. Names should carry the *what*.
- Don't reference the current task or PR in comments. That belongs in commit messages and PR descriptions.

## Tests

- Every new public function or behavior gets a unit test.
- Performance changes get a measurement added to (or updated in) `tests/benchmarks/tmnn_runtime_benchmarks.cpp`. Note the device and command in the PR description; we are not publishing a numbers board until a reproducibility story is ready.
- Tests are organized by directory mirroring `src/tiny_metal_nn/` (e.g., `tests/unit/tmnn/runtime/`, `tests/unit/tmnn/kernels/`, `tests/unit/tmnn/extension/`).
- gtest is the test framework. Use `EXPECT_*` for soft assertions and `ASSERT_*` for assertions that must hold for the test to make sense at all.

Numerics tests use bit-exact comparisons for the deterministic paths and tolerance bounds for the FP-stochastic paths; see existing tests for the convention.

## Commits

- One logical change per commit.
- Imperative subject line, ≤72 characters.
- Body explains *why*, not *what*. The diff shows the what.
- Reference any related `docs/*.md` document in the commit body when the change touches a documented contract (e.g. `CHECKPOINT-CONTRACT.md`, `ERROR-HANDLING.md`).

## Pull requests

Before opening:

1. Local build is green: `cmake --build build && ctest --test-dir build -V`.
2. For changes that touch buffer ownership, lifetimes, or Metal resource
   handling, also confirm the sanitizer build is clean:
   `cmake -S . -B build-asan -DTMNN_ENABLE_SANITIZERS=ON && cmake --build build-asan && ctest --test-dir build-asan`.
3. No leftover `printf` / `std::cout` debug output.
4. New code has tests; modified code's tests still pass.
5. If your change moves a perf number, the new measurement (device, command, before/after) appears in the PR description.

Reviewers will look for:

- Does it match the scope statement at the top of this document?
- Is the change isolated to its stated purpose (no opportunistic refactors riding on top)?
- Does the test coverage match the change's surface area?
- Does it preserve the planner-reasons / numerics-report / checkpoint-contract surfaces? (These are stable contracts; surface changes need an explicit rationale.)

## Architectural decisions

If your contribution changes a load-bearing contract (the network-planner reasons, the kernel-emitter API, the checkpoint format, the error-handling contract, etc.), please update the corresponding document under `docs/` in the same PR — for example, a checkpoint format change should also update [`docs/CHECKPOINT-CONTRACT.md`](docs/CHECKPOINT-CONTRACT.md). The intent is that the public docs always describe the current contract, not last quarter's contract.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)).
