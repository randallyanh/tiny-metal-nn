"""Sphere SDF training — original tinycudann (CUDA) version.

Learns the signed distance function of a unit sphere of radius 0.5 using
the canonical instant-NGP recipe: hash-grid encoding → fully-fused MLP →
MSE loss → Adam optimizer.

This file is the **before** of the tcnn → tmnn migration. The migrated
Apple Silicon equivalent lives next door in
`examples/migrated/sphere_sdf/tmnn/train.py`. See the README for how
they were related.

Runtime requirements (this file):
    - CUDA-capable GPU
    - PyTorch with CUDA build
    - tinycudann (https://github.com/NVlabs/tiny-cuda-nn)

Apple Silicon users: skip this file; run the tmnn variant instead.
"""

from __future__ import annotations

import torch
import tinycudann as tcnn  # type: ignore[import-not-found]

# instant-NGP-style config — small enough to converge in ~50 steps and
# small enough to run on any consumer GPU.
ENCODING_CONFIG = {
    "otype": "HashGrid",
    "n_levels": 4,
    "n_features_per_level": 2,
    "log2_hashmap_size": 14,
    "base_resolution": 16,
    "per_level_scale": 1.5,
}
NETWORK_CONFIG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 16,
    "n_hidden_layers": 1,
}

BATCH_SIZE = 4096
N_STEPS = 50
LEARNING_RATE = 1e-2
SPHERE_RADIUS = 0.5


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=1,
        encoding_config=ENCODING_CONFIG,
        network_config=NETWORK_CONFIG,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for step in range(N_STEPS):
        # Sample uniformly in the [-1, 1]^3 cube. tcnn's HashGrid expects
        # inputs in [0, 1]^3 so we pre-scale.
        positions = torch.rand(BATCH_SIZE, 3, device=device)
        # Target: signed distance to the unit sphere of radius 0.5,
        # centered at (0.5, 0.5, 0.5) so the sphere fits the [0,1]^3 box.
        target = (
            (positions - 0.5).norm(dim=1, keepdim=True) - SPHERE_RADIUS
        )

        output = model(positions)
        loss = ((output - target) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0 or step == N_STEPS - 1:
            print(f"step {step:3d}: loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
