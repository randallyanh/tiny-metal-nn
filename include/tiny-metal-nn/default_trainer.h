#pragma once

/**
 * @file default_trainer.h
 * @brief One-liner factory for complete training + inference.
 *
 * Usage:
 *   auto trainer = tmnn::create_trainer();
 *   auto result  = trainer.training_step(positions, targets, N);
 *   trainer.evaluate(positions, output, N);
 */

#include "tiny-metal-nn/extension/training_adapter.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/trainer.h"

#include <memory>

namespace tmnn {

/// Recommended lightweight preset for the headline create_trainer() path.
[[nodiscard]] HashGridEncoding::Config default_trainer_encoding_config();

/// Recommended network preset paired with default_trainer_encoding_config().
[[nodiscard]] FullyFusedMLP::Config
default_trainer_network_config(
    const HashGridEncoding::Config &enc_cfg = default_trainer_encoding_config());

/// Recommended training preset for interactive default-runtime use.
[[nodiscard]] TrainerConfig default_trainer_config();

/// Non-throwing counterpart to create_trainer(...). Returns `Result<Trainer>`
/// with structured diagnostics instead of raising on construction failure.
[[nodiscard]] Result<Trainer>
try_create_trainer(const HashGridEncoding::Config &enc_cfg =
                       default_trainer_encoding_config(),
                   const FullyFusedMLP::Config &net_cfg =
                       default_trainer_network_config(),
                   const TrainerConfig &train_cfg = default_trainer_config(),
                   std::shared_ptr<MetalContext> ctx = nullptr);

/// Create a complete Trainer from encoding + network + training config.
/// If called without explicit configs, uses the lightweight headline preset above.
/// If ctx is null, creates a new MetalContext.
[[nodiscard]] Trainer
create_trainer(const HashGridEncoding::Config &enc_cfg =
                   default_trainer_encoding_config(),
               const FullyFusedMLP::Config &net_cfg =
                   default_trainer_network_config(),
               const TrainerConfig &train_cfg = default_trainer_config(),
               std::shared_ptr<MetalContext> ctx = nullptr);

/// Create a Trainer from an already-composed tmnn model while keeping the
/// runtime hidden from callers.
[[nodiscard]] Result<Trainer>
try_create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                   const TrainerConfig &train_cfg = {},
                   std::shared_ptr<MetalContext> ctx = nullptr);

/// Throwing convenience wrapper over try_create_trainer(model, ...).
[[nodiscard]] Trainer
create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
               const TrainerConfig &train_cfg = {},
               std::shared_ptr<MetalContext> ctx = nullptr);

/// Create a Trainer from an already-composed tmnn model and explicit loss /
/// optimizer objects.
///
/// The default runtime lowers installed built-in loss descriptors
/// (`L2Loss`/`L1Loss`/`HuberLoss`) into `TrainerConfig` before runtime creation.
/// Optimizer behavior is still largely derived from `TrainerConfig` and runtime
/// policy. The supplied objects are preserved on the `Trainer` surface for
/// inspection and learning-rate synchronization, but this overload should not be
/// read as a fully generic runtime-pluggable loss / optimizer seam.
[[nodiscard]] Result<Trainer>
try_create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                   std::shared_ptr<Loss> loss,
                   std::shared_ptr<Optimizer> optimizer,
                   const TrainerConfig &train_cfg = {},
                   std::shared_ptr<MetalContext> ctx = nullptr);

/// Throwing convenience wrapper over try_create_trainer(model, loss, optimizer, ...).
[[nodiscard]] Trainer
create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
               std::shared_ptr<Loss> loss,
               std::shared_ptr<Optimizer> optimizer,
               const TrainerConfig &train_cfg = {},
               std::shared_ptr<MetalContext> ctx = nullptr);

/// Create a Trainer from a TrainingAdapter + model.
///
/// This constructor validates the adapter schema against the model, validates
/// adapter-declared compile preferences against that schema, and lowers the
/// adapter's declarative loss_config() into `TrainerConfig`.
///
/// In the default runtime this is still a *narrow* seam: full adapter lifecycle
/// callbacks (batch packing, train param filling, config tails, Adam schedules,
/// result metrics) remain the responsibility of adapter-native runtimes.
[[nodiscard]] Result<Trainer>
try_create_trainer_with_adapter(const extension::TrainingAdapter &adapter,
                                std::shared_ptr<NetworkWithInputEncoding> model,
                                const TrainerConfig &train_cfg = {},
                                std::shared_ptr<MetalContext> ctx = nullptr);

/// Throwing convenience wrapper over try_create_trainer_with_adapter(...).
[[nodiscard]] Trainer
create_trainer_with_adapter(const extension::TrainingAdapter &adapter,
                            std::shared_ptr<NetworkWithInputEncoding> model,
                            const TrainerConfig &train_cfg = {},
                            std::shared_ptr<MetalContext> ctx = nullptr);

} // namespace tmnn
