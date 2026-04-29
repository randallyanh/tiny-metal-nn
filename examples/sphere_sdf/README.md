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

## Cross-implementation comparison (status)

The 006 v2 §4.3 contract calls for CI to run *both* implementations on
the same dataset and assert their final losses agree within 5×.

In practice the tcnn side requires CUDA hardware that Apple Silicon CI
does not have. Until a separate CUDA CI lane is wired up, the
verification is split:

- **CI (Apple Silicon)** — runs the tmnn version, asserts
  `final_loss < 0.01` after 50 steps. Anything above that is treated
  as a P0 regression.
- **Manual / external** — anyone with a CUDA machine can run the tcnn
  version and visually compare the loss curve to the published values
  above. We do not block merges on this comparison today.

The 5×-of-tcnn assertion lands in CI when a CUDA-capable runner is
available (006 v2 §11 stage 11 polish).

## Why this example specifically

- **Small enough for CI** — 50 steps, batch 4096, finishes in ~2 seconds
  on M1 Pro.
- **Real, not toy** — sphere SDF is the textbook hash-grid + MLP
  workload; matches what Nerfstudio / instant-NGP fork code does in
  miniature.
- **Exercises every stage of the binding** — Trainer.from_config,
  training_step (numpy + torch interop, stage 7), the fused loss path,
  and final loss assertion via the binding's float return.

If a contributor breaks any of those layers, this example fails first.
