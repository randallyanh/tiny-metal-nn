# Sphere SDF — tcnn → tmnn migration example

The smallest meaningful instant-NGP-style task: learn the signed
distance to a unit sphere of radius 0.5. Used as the canonical
demonstration that `tools/migrate_tcnn.py` + the Claude Code skill
produce code that actually trains correctly on Apple Silicon.

## Layout

```
sphere_sdf/
├── tcnn/train.py    # original CUDA-only tinycudann version (~70 lines)
├── tmnn/train.py    # Apple Metal port produced by the migration tool
└── README.md        # this file
```

## How the migration was performed

1. Started from `tcnn/train.py` — a typical instant-NGP-style training
   script with hash-grid encoding + fully-fused MLP + manual MSE loss
   + `torch.optim.Adam`.

2. Ran the CLI in `--check` mode to see the diagnostics:

   ```
   $ python tools/migrate_tcnn.py examples/sphere_sdf/tcnn/train.py --check
   ...:6: info: rewrote `import tinycudann as tcnn` to `import tiny_metal_nn as tmnn`
   ...:39: warning: tcnn.NetworkWithInputEncoding(...) does not have a 1-to-1 tmnn equivalent...
   ...:50: warning: training loop with .backward() + .step() detected...
   ```

3. Applied the mechanical rewrites with `--output`:

   ```
   $ python tools/migrate_tcnn.py examples/sphere_sdf/tcnn/train.py \
         --output examples/sphere_sdf/tmnn/train.py.partial
   ```

4. Layered the manual edits the CLI flagged (the WARNING-level
   diagnostics in step 2):

   - Inlined `encoding_config` + `network_config` + the `loss = MSE`
     expression + `torch.optim.Adam(lr=1e-2)` + `BATCH_SIZE` literal
     into a single `CONFIG` dict and called
     `tmnn.Trainer.from_config(CONFIG, n_input=3, n_output=1)` (006 v2
     §10.2 mapping table).
   - Replaced the 5-line training loop body with a single
     `loss = trainer.training_step(positions, target)` call (006 v2
     §10.3).
   - Removed `.to('cuda')` since `tmnn.Trainer` implicitly uses the
     local Metal device.

5. Verified convergence — the tmnn version trains cleanly:

   ```
   step   0: loss = 0.130579
   step  10: loss = 0.023629
   step  20: loss = 0.007029
   step  30: loss = 0.003719
   step  40: loss = 0.001589
   step  49: loss = 0.001447
   ```

   `tests/python/test_examples_convergence.py` enforces `final_loss <
   0.01` in CI so accidental regressions cannot land.

## Running

### tmnn version (Apple Silicon, the migrated form)

From the repo root, with the dev venv active and `tiny_metal_nn` built
(see `tests/python/README.md` for the venv setup):

```
.venv/bin/python examples/sphere_sdf/tmnn/train.py
```

### tcnn version (CUDA only — the original)

This file is preserved verbatim as the migration's "before". You only
need to run it if you want to verify the cross-implementation
convergence claim on a CUDA box. Apple Silicon users can ignore it.

```
# on a CUDA-capable machine, with tinycudann installed
python examples/sphere_sdf/tcnn/train.py
```

## Reference metrics

[`reference_metrics.json`](reference_metrics.json) records the numbers
this sample produces on a known machine (M1 Pro, 32 GB, macOS 15) and
the perf-regression gates CI uses. Excerpt:

| Metric | Value | CI gate |
|---|---|---|
| step 49 loss | 0.001447 | `< 0.01` |
| step warm median time | 1.40 ms | `< 5 ms` |
| host RSS delta | 188 MB | `< 350 MB` |
| trainer construct | 96 ms | (informational) |
| total params (fp32) | 131,233 | (informational) |

Update [`reference_metrics.json`](reference_metrics.json) when an
intentional change shifts a number; do **not** raise a threshold to
make a failing test pass — find the regression first.

## Discoveries log (what this sample taught us about tmnn)

This sample lives in the **smoke-test tier** of the network-experiment
doctrine — it confirms the v1.0 binding pipeline is wired correctly and
serves as a stable regression anchor. It is **not** a discovery tool.
Larger samples in the planned set (`bunny_sdf/`, `dragon_sdf/`,
`nerf_lego/`, `neus_surface/`) put more pressure on tmnn's hot paths and
are where genuine capability / performance / functional gaps surface.

What `sphere_sdf` validated, end-to-end:

- the v1.0 Python binding wires correctly: `Trainer.from_config` →
  numpy / torch CPU tensor input → fused `training_step` → `inference`
  → `close` / `__exit__`
- the libcst-based migration CLI (`tools/migrate_tcnn.py`) produces code
  that trains correctly after the documented manual-finish step (drop
  `device = torch.device("cuda")`, drop `.to(device)`, change
  `loss.item()` → `loss`)
- the convergence guard contract (`final_loss < threshold`) is a stable
  CI pattern across tmnn versions

What `sphere_sdf` did **not** drive:

The tmnn improvements that involve sphere-SDF data — P5 init's 460×
convergence speedup, GPU Philox cold-start drop from ~240 ms to
sub-millisecond — were discovered via the dedicated
`--init-convergence-ablation` benchmark in
`tests/benchmarks/tmnn_runtime_benchmarks.cpp`, not via this sample.
sphere_sdf provided the input function for that ablation, not the
diagnostic instrument.

For tmnn diagnostic value, run the larger samples once they ship.

## Cross-implementation comparison (intentionally not done)

The original v1.0 plan called for CI to run both implementations on the
same dataset and assert their final losses agree within 5×. This is no
longer the plan: matched-hardware vs-tcnn benchmarks were dropped from
the roadmap (see internal doctrine for the rationale — the short
version is "different hardware tiers make any cross comparison noisy
enough that the marketing value is below the maintenance cost").

The `tcnn/train.py` file is preserved verbatim as documentation of
"the migration's `before`" — it is not run by CI. Apple Silicon users
can ignore it. CUDA-capable readers who want to run it for personal
verification:

```
python examples/sphere_sdf/tcnn/train.py
```

## Why this example specifically

- **Small enough for CI** — 50 steps, batch 4096, finishes in ~2 seconds
  on M1 Pro.
- **Exercises every layer of the binding** — `Trainer.from_config`,
  `training_step` (numpy + torch interop), the fused loss path, the
  inference path, the close / context manager lifecycle.
- **Stable regression anchor** — if a contributor breaks any of those
  layers, this example fails first via the CI thresholds in
  [`reference_metrics.json`](reference_metrics.json).
