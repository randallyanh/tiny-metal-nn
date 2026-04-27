# tmnn Checkpoint / Reload Contract

This document describes the public optimizer-state checkpoint contract for
`tiny-metal-nn`.

## Scope

The checkpoint surfaces described here are **optimizer-state checkpoints**.

They preserve:

- the logical optimizer step
- Adam first-moment state for hash-grid weights
- Adam second-moment state for hash-grid weights
- Adam first-moment state for MLP weights
- Adam second-moment state for MLP weights

They do **not** preserve:

- model weights
- network/encoding descriptors
- trainer hyperparameters or policy overrides
- autotune manifests or runtime cache state

Ownership is explicit:

- model weights stay owned by the caller-facing weight object
- optimizer checkpoints are a separate payload that must be paired with the
  matching weights and configuration by the caller

## Canonical transport: `tmnn::OptimizerStateBlob`

The canonical transport is `tmnn::OptimizerStateBlob`, exported and imported
through `Trainer::export_optimizer_state()` /
`Trainer::import_optimizer_state(...)`.

Fields:

- `version`
- `step`
- `payload`

Contract:

- `version` must equal `tmnn::kOptimizerStateBlobVersion`
- readers require an exact version match
- `payload` is version-defined and opaque at the public API level
- trailing bytes are rejected

This is the stable binary transport for runtime-owned optimizer state.

## Practical usage

To restore a full training session safely, a caller must restore both:

1. the model weights through the weight object's own save/load path
2. the optimizer checkpoint through `OptimizerStateBlob`

```cpp
auto trainer = tmnn::create_trainer(...);
// ... training proceeds ...

tmnn::OptimizerStateBlob blob = trainer.export_optimizer_state();
// persist `blob` alongside the weight object's own checkpoint

// later, in a fresh process:
auto restored_trainer = tmnn::create_trainer(...);
restored_trainer.import_optimizer_state(blob);
// continue training from the restored optimizer state
```

Checkpoint/reload is only defined for compatible model topology and optimizer
state layout. A checkpoint is **not** a generic cross-model migration artifact.

## Versioning policy

- `tmnn::kOptimizerStateBlobVersion` is bumped on any payload-layout change
- readers require an exact match against their compiled-in version constant
- there is no automatic backward-compatibility path; callers wishing to
  migrate state across an incompatible version bump must do so explicitly

This intentionally trades migration ergonomics for a small, auditable surface.
