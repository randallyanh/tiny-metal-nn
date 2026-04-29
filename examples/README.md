# examples/

This directory is reserved for end-to-end examples that go beyond the single
flagship runnable sample at
[`../samples/mlp_learning_an_image.cpp`](../samples/mlp_learning_an_image.cpp).

The flagship sample is intentionally kept under `samples/` (matching the
upstream tiny-cuda-nn layout, so a tcnn-based codebase can be ported with
fewer directory renames). Larger end-to-end workloads — ones that don't fit
the "single small `.cpp` file" shape — will land here instead.

## Current

- [`sphere_sdf/`](sphere_sdf/) — analytical signed-distance-field fit on
  a unit sphere. The smallest meaningful instant-NGP-style task; serves
  as the canonical Python-binding sample and the CI convergence guard
  (`tests/python/test_examples_convergence.py`). Includes a `tcnn/`
  reference implementation (CUDA only) alongside the runnable `tmnn/`
  Apple Silicon version.

## Planned

Tracked in [`../STATUS.md`](../STATUS.md) under "What does not work yet":

- `nerf_synthetic/` — NeRF synthetic-scene training with HashGrid +
  FullyFusedMLP.
- `hash_grid_sdf_fitting/` — Signed-distance-field fitting on a
  multi-resolution hash grid (extension of `sphere_sdf/` to a real mesh).

Both are open work; they are not implemented yet. Until they land, the
canonical runnable example pair is `sphere_sdf/` here plus
`samples/mlp_learning_an_image.cpp` for the C++ side.
