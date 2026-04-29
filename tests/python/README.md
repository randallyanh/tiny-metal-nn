# Python torture suite

Lifetime / GIL / leak / dtype safety tests for the `tiny_metal_nn` Python
binding. Stage 3 of `docs/know-how/006-python-binding-design.md` v2 § 11.

## Status (2026-04-28)

- **Skeleton landed.** 5 placeholder tests, all `pytest.skip` until the v1.0
  Python binding ships (stage 4 of 006 v2 § 11).
- **Binding does not exist yet.** The `tmnn` fixture in `conftest.py` skips
  every test that depends on it.

## Setup (one-time)

The torture suite runs in a project-local venv to keep deps reproducible
and to avoid polluting the user's global / mambaforge environment.

```bash
# from the repo root
python3.13 -m venv .venv

# scikit-build-core is the build backend; install it first so the
# editable install can use --no-build-isolation (faster, predictable
# vcpkg pickup).
.venv/bin/pip install scikit-build-core pybind11

# Build + install the binding in editable mode. VCPKG_ROOT must point
# at a vcpkg checkout (see CONTRIBUTING.md).
VCPKG_ROOT=/path/to/vcpkg .venv/bin/pip install -e ".[dev]" --no-build-isolation
```

Python **3.13** is the recommended dev version (latest stable 2026-04). The
project supports Python `>=3.10` for end-user wheels; CI tests the matrix
3.10 / 3.11 / 3.12 / 3.13.

After the editable install, edits to `src/python/tmnn_pybind.cpp` (or any
C++ source the binding pulls in) trigger an auto-rebuild on the next
`import tiny_metal_nn` — no manual `cmake --build` needed.

## Running

```bash
# from the repo root, after venv setup
.venv/bin/python -m pytest -v
```

The full suite is currently 56 tests, ~10s on M1 Pro.

## Layout

- [`conftest.py`](conftest.py) — pytest fixtures (`tmnn`, `baseline_config`)
- [`test_lifetime_torture.py`](test_lifetime_torture.py) — the 5 stage-3
  placeholder tests; reference impls embedded as comments for the stage-4
  binding implementer
- pytest config lives in the repo-root `pyproject.toml` under
  `[tool.pytest.ini_options]` — `testpaths = ["tests/python"]` makes
  `pytest` from the repo root find this suite

## CI integration (TODO)

The repo does not currently have `.github/workflows/`. When CI is set up, the
Python torture suite must:

1. Run on macOS Apple Silicon (Metal coverage requires real device)
2. Run after the C++ build produces a `tiny_metal_nn` Python wheel
3. Treat `SKIPPED` as a soft failure during stage 4 ramp (gradually each
   `pytest.skip` removal must accompany a real assertion that passes)
4. Treat `SKIPPED` as a **hard failure** post-v1.0 (no test should remain
   skipped without an issue link)

See `tests/CMakeLists.txt` for the C++ test harness pattern; the Python
suite should hang off the same `ctest` invocation if practical, or run as
a parallel `pytest` job.

## What lands when

| Test | Unskips at |
|---|---|
| `test_trainer_destruction_no_leak` | stage 4 (binding lifecycle wired) |
| `test_gil_released_during_training_step` | stage 5 (GIL release pattern wired) |
| `test_concurrent_training_step_raises` | stage 5 (thread-safety guard wired) |
| `test_output_tensor_outlives_trainer` | stage 4 (inference returns owned buffer) |
| `test_dtype_fp64_raises` | stage 4 (numpy dtype guard at binding entry) |

The other 7 tests from 007 § 6.1 (KeyboardInterrupt, ref-cycle, Metal memory
baseline, async inflight, non-contiguous input, long-run host leak, close
idempotent) ship in stage 11 (final polish).
