# tiny-metal-nn

`tiny-metal-nn` (`tmnn`) is a Metal-native, [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)-inspired C++ library for hash-grid + MLP neural-field training and inference on Apple GPUs.

`tiny-cuda-nn` is widely used in the CUDA ecosystem (instant-ngp, NeuS, Plenoxels, NerfAcc, and others) but does not run on Apple Silicon. `tmnn` is an attempt to provide a runtime in that style on Metal — fully fused MLPs, hash-grid encodings, and JSON config compatible with tcnn — without trying to be a general ML framework, a PyTorch / MLX replacement, or a generic GPU compute substrate.

See [`STATUS.md`](STATUS.md) for what is and is not in the box today, and what we explicitly do not yet claim.

## When tmnn might fit

- You are on Apple Silicon and want a small-network runtime with an API surface close to tcnn (JSON config, fused MLP families, an image-fitting sample), so that a tcnn-based codebase can be ported with relatively small diffs.
- You want explicit family selection (`FullyFusedMetal` / `TiledMetal` / `SafeDebugMetal`), inspectable planner reasons, manifest-backed kernel prewarm, numerics telemetry, and a frozen optimizer-checkpoint contract — operational surfaces we have intentionally exposed.

For a general Apple-native tensor framework, [MLX](https://github.com/ml-explore/mlx) is broader and more mature. On NVIDIA, [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) is the natural choice. A side-by-side comparison, including what we have not yet measured, is in [`docs/VS-MLX-AND-TCNN.md`](docs/VS-MLX-AND-TCNN.md).

## Status

- ~17K LOC C++ source; ~10.5K LOC tests
- `samples/mlp_learning_an_image.cpp` — flagship runnable sample, deliberately matches tiny-cuda-nn's filename and target shape so migration diffs stay small
- Pre-1.0. The v0.1.0 tag is pending external comparison benchmarks (vs MLX, vs tcnn) and at least two more flagship samples (NeRF synthetic, hash-grid SDF fitting). See [`STATUS.md`](STATUS.md) for the current honest scope and the "Roadmap" section below.

## Architecture

```
JSON config → factory_json::canonicalize_model_config
  → NetworkPlan (planner) → kernel selection + prewarm
  → MetalContext (Metal device + pipeline registry + batch pool)
  → fully-fused or tiled MLP training/inference loop
  → checkpoint contract (frozen, optimizer state portable across versions)
```

Targets:

| Target | Purpose |
|---|---|
| `tiny_metal_nn_core` | header-only public API |
| `tiny_metal_nn_runtime` | Metal runtime (MetalContext, pipeline registry, batch pool, training-step lifecycle) |
| `tiny_metal_nn_kernels` | MSL kernel generation (KernelSpec, KernelCompiler, MLPKernelEmitter) |
| `tiny_metal_nn_extensions` | built-in adapters (DNL, RMHE, 4D, standard SDF, multi-output MLP) |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

## Build (C++)

`tmnn` uses [vcpkg](https://github.com/microsoft/vcpkg) for `nlohmann_json` and
`gtest`. Either place a vcpkg checkout at `./.deps/vcpkg`, or point
`VCPKG_ROOT` at an existing one:

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg
```

Then build:

```bash
git clone <repo-url> tiny-metal-nn
cd tiny-metal-nn
cmake -S . -B build
cmake --build build -j
./build/samples/mlp_learning_an_image                           # flagship sample (built-in config)
./build/samples/mlp_learning_an_image my_config.json            # ...with a custom JSON config
ctest --test-dir build -V                                       # tests
```

## Python binding (optional)

A pybind11 binding ships alongside the C++ library. Install via the
project's build backend (scikit-build-core, which invokes the same CMake
project with `-DTMNN_BUILD_PYTHON_MODULE=ON`):

```bash
python3.13 -m venv .venv
.venv/bin/pip install scikit-build-core pybind11
VCPKG_ROOT=~/vcpkg .venv/bin/pip install -e ".[dev]" --no-build-isolation
.venv/bin/python -c "import tiny_metal_nn as tmnn; print(tmnn.__version__)"
```

The editable install reuses a persistent CMake binary directory, so subsequent
edits to `src/python/` (or any C++ source the binding pulls in) auto-rebuild
on the next `import tiny_metal_nn`. See [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
for a Python "hello world" and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
§ 6 for the binding's design.

For users moving from `tinycudann`, a migration tool
(`tools/migrate_tcnn.py`) handles the mechanical conversion (imports + the
canonical 5-line training-loop body); the harder cases (custom losses,
non-canonical shapes) are flagged with actionable diagnostics. A worked
example pair lives at `examples/migrated/sphere_sdf/`. See
[`docs/TCNN-MIGRATION-GUIDE.md`](docs/TCNN-MIGRATION-GUIDE.md) § 10.

For an instrumented build with AddressSanitizer + UndefinedBehaviorSanitizer
(off by default; consumers must rebuild any downstream code with the same
sanitizer flags):

```bash
cmake -S . -B build-asan -DTMNN_ENABLE_SANITIZERS=ON
cmake --build build-asan -j
ctest --test-dir build-asan -j
```

macOS Apple Silicon is the primary target. Linux / non-Apple builds use a Metal stub (compiles, but GPU-runtime tests skip).

## Performance

A standalone GPU-measured benchmark binary lives at
`tests/benchmarks/tmnn_runtime_benchmarks.cpp`. Build and run it on your
hardware to get real numbers:

```bash
cmake --build build --target tmnn_runtime_benchmarks
./build/tests/tmnn_runtime_benchmarks --smoke    # quick check
./build/tests/tmnn_runtime_benchmarks            # full run
```

The binary emits planner / autotune-search / hot-step / morton-sort medians on
the local Metal device. We are intentionally not publishing pre-measured
"reference numbers" in this README until a same-hardware comparison story
against MLX (architecture-equivalent) and `tiny-cuda-nn` (cross-platform)
lands; see [`docs/VS-MLX-AND-TCNN.md`](docs/VS-MLX-AND-TCNN.md) for the
positioning we can honestly defend today, and [`STATUS.md`](STATUS.md) for
what is still on the roadmap.

## Dependencies

External (vcpkg manifest):

- `nlohmann_json` ≥ 3.11.3 for JSON config + checkpoint serialization
- `gtest` ≥ 1.15.0 (test-only)

Apple frameworks (system, macOS): Metal, Foundation.

## Documentation

- [`STATUS.md`](STATUS.md) — single source of truth for what works and what does not
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — five-minute first program (C++ and Python)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — internal design (C++ runtime + Python binding + migration tooling)
- [`docs/VS-MLX-AND-TCNN.md`](docs/VS-MLX-AND-TCNN.md) — honest comparison with named alternatives
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — benchmark methodology and how to run the benchmark binary
- [`docs/TCNN-MIGRATION-GUIDE.md`](docs/TCNN-MIGRATION-GUIDE.md) — porting tcnn projects to tmnn (C++ + Python)
- [`docs/CHECKPOINT-CONTRACT.md`](docs/CHECKPOINT-CONTRACT.md) — optimizer-state checkpoint format
- [`docs/EXTENSIBILITY-DESIGN.md`](docs/EXTENSIBILITY-DESIGN.md) — extension SDK
- [`docs/ERROR-HANDLING.md`](docs/ERROR-HANDLING.md) — `Result<T>` + `DiagnosticCode` contract

## Roadmap to v0.1.0

The following work is on the path to a stable v0.1.0 tag. Until those land,
expect breaking changes on `main` without deprecation notice:

1. Comparison benchmark vs MLX on the same hardware / same workload, with
   architecture-equivalent models on both sides
2. At least two additional flagship samples (NeRF synthetic, hash-grid SDF
   fitting)
3. CI matrix workflows shipped (`.github/workflows/test.yml`,
   `.github/workflows/release.yml`); pending first run + minutes-budget
   review on a public-repo runner
4. PyPI listing for the Python wheel (workflow gated off until a
   maintainer configures trusted publishing)
5. `MAINTAINERS.md` with response SLA

For the current honest scope ("what works today" vs "what does not work yet"),
see [`STATUS.md`](STATUS.md).

## Related projects

- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) — the upstream model for tmnn's API surface and sample shape; the natural choice on NVIDIA hardware
- [MLX](https://github.com/ml-explore/mlx) — Apple's official ML framework; broader scope, the natural choice for general Apple-native tensor work
- [slangcsg](https://github.com/randallyanh/slangcsg) (currently private) — sibling project consuming `tmnn_core` + `tmnn_runtime` for differentiable CSG / SDF compute

## License

Apache-2.0. See [`LICENSE`](LICENSE).
