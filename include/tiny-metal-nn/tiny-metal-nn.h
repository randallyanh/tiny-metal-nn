#pragma once

/**
 * @file tiny-metal-nn.h
 * @brief Umbrella header for tiny-metal-nn — Metal-native neural network
 *        training and inference for hash-grid + MLP architectures.
 *
 * Provides a tcnn-inspired public surface for the common config-driven C++
 * workflow. It aligns well on JSON configuration, trainer creation, training,
 * and inference, but it does not claim full module-level drop-in parity with
 * every tiny-cuda-nn surface.
 *
 * Common mapping:
 *
 *   tcnn                              tmnn
 *   ──────────────────────────────     ──────────────────────────────
 *   #include <tiny-cuda-nn/cpp_api.h> #include <tiny-metal-nn/tiny-metal-nn.h>
 *
 *   tcnn::cpp::Module                 tmnn::NetworkWithInputEncoding
 *   tcnn::cpp::create_encoding()      tmnn::create_encoding()
 *   tcnn::cpp::create_network()       tmnn::create_network()
 *   tcnn::Trainer                     tmnn::Trainer
 *   tcnn::Loss                        tmnn::Loss
 *   tcnn::Optimizer                   tmnn::Optimizer
 *   (implicit CUDA context)           tmnn::MetalContext::create()
 *
 * Quick start:
 *   auto trainer = tmnn::create_trainer(); // lightweight interactive preset
 *   trainer.training_step(positions, targets, N);
 *   trainer.inference(positions, output, N);
 */

// Device context (Metal equivalent of implicit CUDA context)
#include "tiny-metal-nn/metal_context.h"

// Encoding — hash grid, rotated hash grid
#include "tiny-metal-nn/encoding.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/rotated_hash_grid.h"

// Network — fully-fused MLP
#include "tiny-metal-nn/network.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/network_with_input_encoding.h"

// Loss and optimizer
#include "tiny-metal-nn/loss.h"
#include "tiny-metal-nn/optimizer.h"

// Training and inference
#include "tiny-metal-nn/trainer.h"
#include "tiny-metal-nn/evaluator.h"

// Factory functions (cf. tcnn::cpp::create_encoding/network)
#include "tiny-metal-nn/factory.h"

// Default trainer — one-liner for clone→build→train→evaluate
#include "tiny-metal-nn/default_trainer.h"

// Common types — TrainingStepResult, NumericsReport, etc.
#include "tiny-metal-nn/common.h"
