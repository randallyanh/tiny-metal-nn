# tiny-cuda-nn to tiny-metal-nn migration guide

> Scope: current C++ package surfaces and config portability story  
> Related: [`docs/VS-MLX-AND-TCNN.md`](VS-MLX-AND-TCNN.md),
> [`docs/BENCHMARKS.md`](BENCHMARKS.md), `tests/smoke/smoke_test.cpp`

---

## 1. Goal

This guide is for users who already understand the `tiny-cuda-nn` mental model
and want the shortest path to a working `tiny-metal-nn` port on Apple Metal.

It focuses on the migration surfaces that exist **today**:

- typed builders in `tiny-metal-nn/factory.h`
- `tiny-cuda-nn`-style JSON canonicalization in `tiny-metal-nn/factory_json.h`
- planner / manifest / prewarm APIs in
  `tiny-metal-nn/network_with_input_encoding.h` (returns `NetworkPlan`),
  `tiny-metal-nn/metal_context.h`, and
  `tiny-metal-nn/autotune_manifest.h`

Python onboarding is on the roadmap but not yet implemented; see
[`STATUS.md`](../STATUS.md).

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
- Python onboarding is on the roadmap but not yet implemented

Those are not reasons to block the first C++ port. They are reasons to keep the
scope of the first port explicit.
