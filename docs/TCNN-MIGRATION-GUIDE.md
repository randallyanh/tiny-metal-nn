# tiny-cuda-nn to tiny-metal-nn migration guide

> Scope: current C++ package surfaces and config portability story  
> Related: [`docs/VS-MLX-AND-TCNN.md`](VS-MLX-AND-TCNN.md),
> [`docs/BENCHMARKS.md`](BENCHMARKS.md), `tests/smoke/smoke_test.cpp`

---

## 1. Goal

This guide is for users who already understand the `tiny-cuda-nn` mental model
and want the shortest path to a working `tiny-metal-nn` port on Apple Metal.

The guide has two halves:

- **§§ 2–9 — C++ migration**, the surfaces that exist today: typed builders in
  `tiny-metal-nn/factory.h`, `tiny-cuda-nn`-style JSON canonicalization in
  `tiny-metal-nn/factory_json.h`, planner / manifest / prewarm APIs in
  `tiny-metal-nn/network_with_input_encoding.h` (returns `NetworkPlan`),
  `tiny-metal-nn/metal_context.h`, and `tiny-metal-nn/autotune_manifest.h`.
- **§ 10 — Python migration**, the planned v1.0 deliverable: a three-piece
  tooling story (CLI translator, Claude Code skill, CI-verified migration
  examples) that converts tcnn Python code to tmnn Python code. The Python
  binding itself is shipping in v1.0; the migration tooling lands alongside it.
  Design contract: [`know-how/006-python-binding-design.md`](know-how/006-python-binding-design.md).

---

## 2. Fast mental model

| `tiny-cuda-nn` concept | `tiny-metal-nn` concept today | Notes |
|------------------------|-------------------------------|-------|
| Encoding config | `Encoding` + `create_encoding(...)` / `create_encoding_from_json(...)` | Current JSON bridge accepts the common HashGrid / rotated HashGrid shapes |
| Network config | `Network` + `create_network(...)` / `create_network_from_json(...)` | Current JSON bridge materializes `FullyFusedMLP` |
| Combined model | `NetworkWithInputEncoding` | Public composition surface for planner/runtime decisions |
| Runtime choice | `NetworkPlan` | Explicitly surfaces chosen family, candidates, and fallback reasons |
| Device/runtime root | `MetalContext` | Owns device capabilities, caches, and in-memory autotune state |
| Warmed selection cache | `AutotuneManifest` | Save/load/prewarm contract for reproducible hot-path reuse |
| Tensor view into data | `TensorRef` | Non-owning typed view; ownership stays with the caller/runtime |
| Optimizer checkpoint blob | `OptimizerStateBlob` | Optimizer-only transport; weights/config are caller-owned |

The important difference is that tmnn makes planner choice and cache reuse part
of the public story instead of leaving them as implicit runtime behavior.

---

## 3. The easiest port-first path

Start by reusing the `tiny-cuda-nn`-style config shape instead of rewriting
everything into typed C++ immediately.

```cpp
#include "tiny-metal-nn/factory_json.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/network_with_input_encoding.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

json enc = {
    {"otype", "HashGrid"},
    {"n_levels", 16},
    {"n_features_per_level", 2},
    {"log2_hashmap_size", 19},
    {"base_resolution", 16.0f},
    {"per_level_scale", 1.447f},
};

json net = {
    {"otype", "FullyFusedMLP"},
    {"n_neurons", 64},
    {"n_hidden_layers", 2},
    {"activation", "ReLU"},
    {"output_activation", "None"},
};

auto model =
    tmnn::create_network_with_input_encoding_from_json(3, 1, enc, net);

auto ctx = tmnn::MetalContext::create();
tmnn::NetworkFactoryOptions options;
options.metal_context = ctx;

tmnn::NetworkPlan plan = model->plan(options);
```

That already gives you:

- canonicalized tmnn descriptors
- an explicit selected family
- fallback reasons if fused execution is not eligible
- manifest-backed reuse through the attached `MetalContext`

---

## 4. JSON field mapping that works today

### 4.1 Encoding fields

The checked-in bridge currently accepts these encoding keys:

| Incoming key | Current status |
|--------------|----------------|
| `otype` | Supports `HashGrid`, alias `MultiresolutionHashGrid`, `RotatedMHE`, alias `RotatedMultiresHashGrid` |
| `n_levels` | Supported |
| `n_features_per_level` | Supported |
| `log2_hashmap_size` | Supported |
| `base_resolution` | Supported |
| `per_level_scale` | Supported |
| `interpolation` | Only `Linear` is currently accepted |

### 4.2 Network fields

The checked-in bridge currently accepts these network keys:

| Incoming key | Current status |
|--------------|----------------|
| `otype` | Supports `FullyFusedMLP` |
| `n_neurons` | Supported |
| `n_hidden_layers` | Supported |
| `activation` | Only `ReLU` is currently accepted |
| `output_activation` | `Linear` is normalized to `None`; `None` is supported |
| `training` | Parsed as part of canonical diagnostics, but not a separate full runtime builder contract |

### 4.3 Unsupported / not-yet-public bridge cases

Be explicit about these current limits:

- `TiledMLP` JSON canonicalization is not the current public bridge path
- unsupported interpolation/activation values fail with diagnostics
- the guide here is for the currently landed C++ surface, not the future Python
  wrapper

If your old config is close to the common HashGrid + fully fused MLP shape, the
port is usually a config translation problem rather than a rewrite problem.

---

## 5. Planner and manifest are part of the port

`tiny-cuda-nn` users often think first about the model shape. In tmnn, you
should also think about the execution family as part of the public contract.

The relevant surfaces are:

- `NetworkPlan`
- `MetalContext`
- `AutotuneManifest`

Minimal manifest flow:

```cpp
auto ctx = tmnn::MetalContext::create();
tmnn::NetworkFactoryOptions options;
options.metal_context = ctx;

auto first = model->plan(options);
auto manifest = ctx->snapshot_autotune_manifest();
tmnn::save_autotune_manifest("tmnn-autotune.json", manifest);

auto next_ctx = tmnn::MetalContext::create();
next_ctx->prewarm_autotune_manifest(
    tmnn::load_autotune_manifest("tmnn-autotune.json"));
options.metal_context = next_ctx;

auto second = model->plan(options);
```

The contract you can rely on:

- `first.planner_fingerprint` identifies the planned model/device/options shape
- `second.from_autotune_manifest` tells you hot-path reuse actually happened
- `candidate_families` and `reasons` tell you why the selected family was or was
  not eligible

See sibling `../tiny-metal-nn/tests/smoke/smoke_test.cpp` for a minimal
external consumer that exercises this flow through the installed package.

---

## 6. Ownership rules to keep in mind

### 6.1 Tensor ownership

`TensorRef` is a **non-owning view**:

- it points at caller/runtime-owned memory
- it carries shape + precision metadata
- it is not the owner of the buffer

If you are porting code that assumed a framework-owned tensor object, be careful
to keep buffer ownership explicit.

### 6.2 Runtime ownership

`MetalContext` is the runtime root for:

- device capabilities
- caches
- in-memory autotune decision state
- runtime policy / stats

Share it when you want warmed reuse within one execution graph.

### 6.3 Checkpoint ownership

`OptimizerStateBlob` is intentionally optimizer-only:

- it includes the logical optimizer step and opaque optimizer payload
- it does **not** include weights
- it does **not** include descriptors or trainer config

When you port checkpoint logic, keep weights/config checkpointing as caller-owned
state and pair it with the optimizer blob on restore.

---

## 7. Training modes today

There are two practical ways to approach a migration today.

### 7.1 Port-first mode

Use the installed/public config and planning surfaces first:

- `factory_json.h`
- `NetworkWithInputEncoding`
- `MetalContext`
- `NetworkPlan`
- `AutotuneManifest`

This is a reasonable first step when the immediate goal is:

- config translation
- planner visibility
- reproducible hot-path behavior
- checkpoint / manifest integration planning

### 7.2 Full GPU training runtime mode

The public object model for training is in:

- `include/tiny-metal-nn/trainer.h`

The concrete GPU-backed runtime ships in the `tiny_metal_nn_runtime` CMake
target. From a downstream `find_package(tiny_metal_nn CONFIG REQUIRED)`, link
against the alias:

```cmake
target_link_libraries(my_app PRIVATE tiny_metal_nn::tiny_metal_nn_runtime)
```

That gives you `MetalContext`, the planner, the autotune manifest surfaces,
and the default `Trainer` runtime in a single linked library — no external
runtime package required.

---

## 8. Recommended port order

1. Port the config through `factory_json.h`.
2. Confirm the descriptor shape with `NetworkWithInputEncoding`.
3. Attach a `MetalContext` and inspect the `NetworkPlan`.
4. Save/load a manifest and prove `from_autotune_manifest` on the hot path.
5. Only then wire the full GPU training runtime if you need step execution and
   optimizer state export immediately.

This order keeps the first port small and auditable.

---

## 9. Current limitations worth calling out

This guide is honest only if it also lists what remains open:

- matched-hardware comparison against native `tiny-cuda-nn` is not published in
  this repo yet
- the cross-device Apple validation matrix is still active work
- Python migration tooling (§ 10) is in design contract; implementation lands
  with the v1.0 Python binding

Those are not reasons to block the first C++ port. They are reasons to keep the
scope of the first port explicit.

---

## 10. Python migration (planned for v1.0)

> Status: design contract finalized 2026-04-28. Implementation is part of the
> v1.0 Python binding milestone. See
> [`know-how/006-python-binding-design.md`](know-how/006-python-binding-design.md)
> for the full design and
> [`know-how/005-tcnn-compatibility-strategy.md`](know-how/005-tcnn-compatibility-strategy.md)
> for the strategic decision record.

### 10.1 The architectural shift

`tiny-cuda-nn` and `tiny-metal-nn` use different training-loop abstractions:

```python
# tcnn pattern: model.forward + user-owned loss + user-owned optimizer
output = model(input)                         # tcnn.NetworkWithInputEncoding(...)
loss   = ((output - target) ** 2).mean()      # PyTorch
loss.backward()                               # PyTorch autograd
optimizer.step()                              # PyTorch optim.Adam
optimizer.zero_grad()
```

```python
# tmnn pattern: trainer.training_step (forward + backward + optimizer fused)
loss = trainer.training_step(input, target)
```

This is the central, non-mechanical part of the migration: tmnn's fused
`training_step` runs forward, backward, and the Adam step in a single Metal
command buffer, which is where the Apple-side speedup comes from. Migrating
into tmnn means moving the loss + optimizer choice into `Trainer.from_config`
rather than the user's training loop.

### 10.2 Three migration paths

| Path | When to use | Status |
|------|-------------|--------|
| **CLI**: `tools/migrate_tcnn.py` | Mechanical translation of single files / directories. Best for code that uses standard tcnn idioms. | v1.0 deliverable |
| **Claude Code skill**: `.claude/skills/tcnn-to-tmnn.md` | AI-assisted migration that handles edge cases the CLI cannot (custom losses, non-trivial encoding configs, project-wide refactors). Reads the same translation rules as the CLI. | v1.0 deliverable |
| **Manual** | Reading this guide + the field-level mapping table in `know-how/006-python-binding-design.md` § 10. Useful when you want to understand each rule. | Always |

All three paths share one source of truth: `tools/migrate_rules.py`. The CLI
and skill consume it directly; the manual path is its README-level summary.

### 10.3 Workflow (planned)

```bash
# 1. Inspect what would change, no files written
$ tmnn-migrate train.py --check --diff

# 2. Apply the migration
$ tmnn-migrate train.py --output train_tmnn.py

# 3. Run the migrated code
$ python train_tmnn.py
```

Exit codes:

- `0` — every translation rule was applied; the migrated file is ready
- `1` — some segments need human review (custom loss, fp16, custom encoding,
  multi-GPU). Diagnostics list each unhandled segment with file:line and
  guidance
- `2` — input could not be parsed

### 10.4 Quick reference — the mappings most likely to apply

For the complete mapping see
[`know-how/006-python-binding-design.md`](know-how/006-python-binding-design.md)
§ 10. The condensed view:

| tcnn | tmnn | Notes |
|------|------|-------|
| `import tinycudann as tcnn` | `import tiny_metal_nn as tmnn` | mechanical |
| `tcnn.NetworkWithInputEncoding(n_in, n_out, enc, net).to('cuda')` | `tmnn.Trainer.from_config({"n_input": n_in, "n_output": n_out, "encoding": enc, "network": net, "loss": ..., "optimizer": ...})` | loss + optimizer move into config; `'cuda'` → `'mps'` (or omit) |
| `output = model(input); loss = ((output-target)**2).mean(); loss.backward(); optimizer.step()` | `loss = trainer.training_step(input, target)` | fused |
| Encoding `otype: HashGrid` and standard fields (`n_levels`, `n_features_per_level`, `log2_hashmap_size`, `base_resolution`, `per_level_scale`, `interpolation: Linear`) | identical | direct copy |
| Network `otype: FullyFusedMLP`, `n_neurons`, `n_hidden_layers`, `activation: ReLU` | identical | direct copy |
| `otype: CutlassMLP` | `otype: FullyFusedMLP` | auto-mapped; CutlassMLP has no Metal equivalent |
| `output_activation: None` | `output_activation: Linear` | alias normalized |
| Optimizer `otype: Adam` with `learning_rate`, `beta1`, `beta2`, `epsilon`, `l2_reg` | identical | direct copy |
| Loss `otype: L2`, `L1`, `Huber` (and `RelativeL2`, `SmoothL1`) | identical | direct copy |

### 10.5 What the migration tool will warn about (human review needed)

| Pattern | Reason |
|---------|--------|
| fp16 / `torch.cuda.amp` / `loss_scale` | tmnn v1.0 is fp32-only; fp16 is a v1.x+ roadmap item |
| Multi-GPU / DDP | Apple Silicon is single-GPU; the migration tool removes the multi-GPU init code and warns |
| Custom loss expressions (anything beyond MSE / L1 / Huber) | The fused `training_step` only supports built-in losses. The tool suggests either: (A) substitute the closest built-in, or (B) drop to the non-fused `trainer.forward(...) + manual backward` path |
| Custom encodings (`otype` not in tmnn's list, e.g. `SphericalHarmonics`, `Frequency`, `OneBlob`) | No automatic equivalent; the tool suggests workarounds (use `MlpInit::Uniform` with explicit bound, or keep the encoding in PyTorch and feed the encoded features into tmnn) |
| `.to('cuda')` / `'cuda:0'` | Rewritten to `.to('mps')` or removed; warned because performance / numerics may differ from the original CUDA report |

### 10.6 What's intentionally **not** in scope for v1.0

- `tiny_metal_nn.nn.NetworkWithInputEncoding(torch.nn.Module)` — the
  `nn.Module` wrapping path. The migration tool replaces this need; if a
  concrete user case shows it cannot, the path becomes a v1.x+ candidate. See
  `know-how/006-python-binding-design.md` v2 § 5.1 and § 12.
- A `tiny_metal_nn.compat.tcnn` shim namespace that lets `import tinycudann`
  keep working. The strategic record (`know-how/005-tcnn-compatibility-strategy.md`
  v2 Layer C) explains why this was rejected in favour of the migration tool.
- Zero-copy MPS interop (`torch.Tensor` on MPS handed straight into the tmnn
  forward kernel without a CPU bounce). The Q3 spike (2026-04-28) confirmed it
  is feasible and stable across PyTorch ≥ 2.1; it ships in v1.x+, not v1.0.
  v1.0 inputs go through a CPU staging copy.
