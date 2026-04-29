"""Sphere SDF training — migrated tiny-metal-nn (Apple Metal) version.

Drop-in equivalent of `../tcnn/train.py`. Same dataset, same optimizer
shape, same step count — produced by running

    python tools/migrate_tcnn.py examples/migrated/sphere_sdf/tcnn/train.py \\
        --output examples/migrated/sphere_sdf/tmnn/train.py

then layering the manual edits the migration tool flagged (see the
parent README for the exact diff and rationale per change).

Runtime requirements:
    - macOS Apple Silicon
    - tiny_metal_nn built and importable (see tests/python/README.md
      for the dev venv setup)
    - PyTorch (any 2.x; only used here for tensor I/O — not training)
"""

from __future__ import annotations

import torch

import tiny_metal_nn as tmnn

CONFIG = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "log2_hashmap_size": 14,
        "base_resolution": 16.0,
        "per_level_scale": 1.5,
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 1,
    },
    # The next three sections moved into from_config from the tcnn
    # training-loop scaffolding (loss expression + torch.optim.Adam +
    # batch-size literal). 006 v2 § 5.1 explains why fusing them is the
    # core perf advantage on Metal.
    "loss": {"otype": "L2"},
    "optimizer": {"otype": "Adam", "learning_rate": 1e-2},
    "batch_size": 4096,
}

BATCH_SIZE = 4096
N_STEPS = 50
SPHERE_RADIUS = 0.5


def run(verbose: bool = True) -> list[float]:
    """Train the sphere-SDF network for `N_STEPS` and return the loss
    history. Used by both ``main()`` and the convergence CI test."""
    torch.manual_seed(42)
    losses: list[float] = []

    with tmnn.Trainer.from_config(CONFIG, n_input=3, n_output=1) as trainer:
        for step in range(N_STEPS):
            # Sample in [0, 1]^3 just like the tcnn version. tmnn accepts
            # torch CPU float32 tensors zero-copy via __array__ (stage 7).
            positions = torch.rand(BATCH_SIZE, 3, dtype=torch.float32)
            target = (
                (positions - 0.5).norm(dim=1, keepdim=True) - SPHERE_RADIUS
            ).to(torch.float32)

            # Single fused step replaces the 5-line tcnn pattern
            # (forward + MSE + backward + step + zero_grad).
            loss = trainer.training_step(positions, target)
            losses.append(loss)

            if verbose and (step % 10 == 0 or step == N_STEPS - 1):
                print(f"step {step:3d}: loss = {loss:.6f}")

    return losses


def main() -> None:
    run(verbose=True)


if __name__ == "__main__":
    main()
