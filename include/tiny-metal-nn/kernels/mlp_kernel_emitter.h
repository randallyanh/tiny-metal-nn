#pragma once

/**
 * @file mlp_kernel_emitter.h
 * @brief MLPKernelEmitter: generates specialized MSL kernel source from KernelSpec.
 *
 * Replaces the 20+ static MSL kernel headers with a single code generator.
 * All architecture dimensions (hidden_dim, num_layers, etc.) are baked as
 * integer constants in the generated code, matching tiny-cuda-nn's template
 * specialization approach.
 */

#include "tiny-metal-nn/kernels/kernel_spec.h"

#include <sstream>
#include <string>

namespace tmnn {

class MLPKernelEmitter {
public:
  /// Generate eval-only kernel (hash encode + MLP forward).
  /// Requires: `spec.validate()` succeeds.
  [[nodiscard]] std::string emitEvalKernel(const KernelSpec& spec);

  /// Generate gradient kernel (hash encode + MLP forward + analytical backward).
  /// Requires: `spec.validate()` succeeds.
  [[nodiscard]] std::string emitGradientKernel(const KernelSpec& spec);

  /// Generate fused eval+gradient kernel (outputs interleaved [sdf, gx, gy, gz]).
  /// Requires: `spec.validate()` succeeds.
  [[nodiscard]] std::string emitEvalGradientKernel(const KernelSpec& spec);

  /// Generate training kernel (forward + loss + backward + hash scatter + TG flush).
  /// Requires: `spec.validate()` succeeds.
  [[nodiscard]] std::string emitTrainKernel(const KernelSpec& spec);

  /// Generate backward kernel with external output gradient.
  /// Uses native fragment composition (not string replacement).
  /// Recomputes forward to recover activations, reads d_output from buffer(8),
  /// then runs MLP backward + hash scatter. No internal loss.
  [[nodiscard]] std::string emitBackwardExternalGradKernel(const KernelSpec& spec);

  /// Options for the unified hash encoding emitter.
  struct HashEncodeOpts {
    bool is_4d = false;
    bool is_rmhe = false;
    bool is_cooperative = false;  // 32 threads, 8 points
    bool fuse_w0 = true;          // emit W0 matmul inline
  };

  /// Generate forward-only kernel for split training path.
  /// Writes network output to buffer(8) = forward_output.
  [[nodiscard]] std::string emitForwardForTrainingKernel(const KernelSpec& spec);

private:
  /// Loss emission mode for emitScalarTrainKernelImpl.
  enum class LossMode { Internal, ExternalGrad, ForwardOnly };

  /// Shared scalar train kernel generator. LossMode controls the loss section:
  /// - Internal: standard emitLoss (L2/L1/Huber)
  /// - ExternalGrad: read d_sdf from buffer(8), no loss computation
  /// - ForwardOnly: write output to buffer(8), no loss/backward/scatter
  [[nodiscard]] std::string emitScalarTrainKernelImpl(
      const KernelSpec& spec, LossMode mode, const char* entry_point);

  // Composable fragments
  void emitPreamble(std::ostringstream& out, const KernelSpec& spec);
  void emitHashFunctions(std::ostringstream& out, const KernelSpec& spec);

  /// Unified hash encoding: parameterized by HashEncodeOpts.
  void emitHashEncodeUnified(std::ostringstream& out, const KernelSpec& spec,
                             const HashEncodeOpts& opts);

  // Convenience wrappers (delegate to emitHashEncodeUnified)
  void emitHashEncode3D(std::ostringstream& out, const KernelSpec& spec);
  void emitHashEncode4D(std::ostringstream& out, const KernelSpec& spec);
  void emitRMHEHashEncode(std::ostringstream& out, const KernelSpec& spec);
  void emitMLPForward(std::ostringstream& out, const KernelSpec& spec);
  void emitMLPBackward(std::ostringstream& out, const KernelSpec& spec);
  void emitLoss(std::ostringstream& out, const KernelSpec& spec);
  /// Helper: emit FP16 loss scaling (* loss_scale or * inv_loss_scale).
  void emitFP16Scale(std::ostringstream& out, const KernelSpec& spec,
                     const char* var, bool forward, int indent = 8);
  void emitHashScatter(std::ostringstream& out, const KernelSpec& spec);
  void emitTGFlush(std::ostringstream& out, const KernelSpec& spec);
  void emitLossReduction(std::ostringstream& out, const KernelSpec& spec);
  void emitProbeReduction(std::ostringstream& out, const KernelSpec& spec);

  // Helper: emit weight offsets as constants for N-layer MLP.
  void emitWeightOffsets(std::ostringstream& out, const KernelSpec& spec);

  // Helper: emit trilinear_feature_grad inline MSL function for gradient kernel.
  void emitTrilinearFeatureGrad3D(std::ostringstream& out);

  /// Shared gradient/fused kernel implementation.
  /// @param include_sdf  true = fused eval+gradient, false = gradient-only.
  [[nodiscard]] std::string emitGradientKernelImpl(const KernelSpec& spec,
                                                    bool include_sdf);

  // SIMD kernel generators (32 threads/TG, 8 points)
  void emitHashEncodeCooperative(std::ostringstream& out, const KernelSpec& spec);
  void emitHashEncodeCoopScalarW0(std::ostringstream& out,
                                  const KernelSpec& spec);
  void emitHashEncodeCoopScalarW0Training(std::ostringstream& out,
                                          const KernelSpec& spec);
  void emitMLPForwardSIMD(std::ostringstream& out, const KernelSpec& spec);
  void emitMLPForwardSIMDHiddenOnly(std::ostringstream& out,
                                    const KernelSpec& spec);
  void emitMLPBackwardSIMD(std::ostringstream& out, const KernelSpec& spec);
};

} // namespace tmnn
