#include "tiny-metal-nn/factory_json.h"
#include "tiny-metal-nn/tiny-metal-nn.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kWidth = 16;
constexpr int kHeight = 16;
constexpr int kPixelCount = kWidth * kHeight;

enum class ExitCode : int {
  Success = 0,
  NoGpu = 2,
  ConfigOpenFailed = 3,
  ConfigParseFailed = 4,
  ReferenceWriteFailed = 5,
  TrainingNonFinite = 6,
  InferenceFailed = 7,
  OutputWriteFailed = 8,
  TrainerCreateFailed = 9,
  UncaughtException = 64,
};

tmnn::json default_config() {
  return {
      {"loss", {{"otype", "L2"}}},
      {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-2f}}},
      {"encoding",
       {{"otype", "HashGrid"},
        {"n_levels", 8},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 14},
        {"base_resolution", 8.0f},
        {"per_level_scale", 1.5f}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", 32},
        {"n_hidden_layers", 2}}},
  };
}

void make_image(std::vector<float> &image) {
  image.resize(static_cast<size_t>(kPixelCount) * 3u);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int index = y * kWidth + x;
      const float u = static_cast<float>(x) / static_cast<float>(kWidth - 1);
      const float v = static_cast<float>(y) / static_cast<float>(kHeight - 1);
      image[static_cast<size_t>(index) * 3u + 0u] = u;
      image[static_cast<size_t>(index) * 3u + 1u] = v;
      image[static_cast<size_t>(index) * 3u + 2u] = 0.5f * (u + v);
    }
  }
}

void make_coords(std::vector<float> &xs_and_ys) {
  xs_and_ys.resize(static_cast<size_t>(kPixelCount) * 3u);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int index = y * kWidth + x;
      const float u = static_cast<float>(x) / static_cast<float>(kWidth - 1);
      const float v = static_cast<float>(y) / static_cast<float>(kHeight - 1);
      xs_and_ys[static_cast<size_t>(index) * 3u + 0u] = u;
      xs_and_ys[static_cast<size_t>(index) * 3u + 1u] = v;
      xs_and_ys[static_cast<size_t>(index) * 3u + 2u] = 0.0f;
    }
  }
}

bool save_image(const std::vector<float> &image, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out)
    return false;

  out << "P6\n" << kWidth << " " << kHeight << "\n255\n";
  for (float value : image) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    const auto byte =
        static_cast<uint8_t>(clamped * 255.0f + 0.5f);
    out.put(static_cast<char>(byte));
  }
  return true;
}

float loss(const std::vector<float> &prediction, const std::vector<float> &target) {
  float sum = 0.0f;
  for (size_t i = 0; i < prediction.size(); ++i) {
    const float diff = prediction[i] - target[i];
    sum += diff * diff;
  }
  return sum / static_cast<float>(prediction.size());
}

int fail(ExitCode code, const std::string &stage, const std::string &message,
         std::optional<tmnn::DiagnosticInfo> diagnostic = std::nullopt) {
  std::cerr << "error_stage=" << stage << '\n';
  std::cerr << "error_message=" << message << '\n';
  if (diagnostic.has_value()) {
    std::cerr << "diagnostic=" << tmnn::format_diagnostic(*diagnostic) << '\n';
  }
  std::cerr << "exit_code=" << static_cast<int>(code) << '\n';
  return static_cast<int>(code);
}

void print_numerics(const tmnn::TrainingStepResult &result) {
  std::cerr << "training_step=" << result.step << '\n';
  std::cerr << "recovery_action="
            << static_cast<uint32_t>(result.recovery_action) << '\n';
  std::cerr << "numerics_reported=" << result.numerics_reported << '\n';
  std::cerr << "has_numerics_anomaly=" << result.has_numerics_anomaly << '\n';
  std::cerr << "finite_forward=" << result.numerics.finite_forward << '\n';
  std::cerr << "finite_backward=" << result.numerics.finite_backward << '\n';
  std::cerr << "finite_update=" << result.numerics.finite_update << '\n';
  std::cerr << "max_abs_activation=" << result.numerics.max_abs_activation << '\n';
  std::cerr << "max_abs_gradient=" << result.numerics.max_abs_gradient << '\n';
  std::cerr << "max_abs_update=" << result.numerics.max_abs_update << '\n';
  std::cerr << "first_bad_layer=" << result.numerics.first_bad_layer << '\n';
}

void print_config_diagnostics(
    const std::vector<tmnn::DiagnosticDetail> &diagnostics) {
  for (size_t i = 0; i < diagnostics.size(); ++i) {
    std::cerr << "config_diagnostic[" << i
              << "]=" << tmnn::format_config_diagnostic(diagnostics[i]) << '\n';
  }
}

bool runtime_inspection_enabled() {
  const char *value = std::getenv("TMNN_SAMPLE_PRINT_RUNTIME_INSPECTION");
  if (!value)
    return false;
  return value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

void print_kernel_inspection(const char *name,
                             const tmnn::TrainerKernelInspection &inspection) {
  std::cout << "kernel." << name << ".available=" << inspection.available << '\n';
  if (!inspection.available)
    return;
  std::cout << "kernel." << name << ".entry_point=" << inspection.entry_point
            << '\n';
  std::cout << "kernel." << name
            << ".requested_simd=" << inspection.requested_simd << '\n';
  std::cout << "kernel." << name
            << ".realized_simd=" << inspection.realized_simd << '\n';
  std::cout << "kernel." << name
            << ".requested_fp16=" << inspection.requested_fp16 << '\n';
  std::cout << "kernel." << name
            << ".realized_fp16=" << inspection.realized_fp16 << '\n';
  std::cout << "kernel." << name
            << ".requested_tg_weight_cache="
            << inspection.requested_tg_weight_cache << '\n';
  std::cout << "kernel." << name
            << ".realized_tg_weight_cache="
            << inspection.realized_tg_weight_cache << '\n';
  std::cout << "kernel." << name
            << ".threadgroup_size=" << inspection.threadgroup_size << '\n';
  std::cout << "kernel." << name
            << ".points_per_threadgroup="
            << inspection.points_per_threadgroup << '\n';
  std::cout << "kernel." << name
            << ".threadgroup_memory_bytes="
            << inspection.threadgroup_memory_bytes << '\n';
}

void print_runtime_inspection(const tmnn::Trainer &trainer) {
  const auto inspection = trainer.inspect_runtime();
  if (!inspection.has_value()) {
    std::cout << "runtime_inspection.available=0\n";
    return;
  }
  std::cout << "runtime_inspection.available=1\n";
  std::cout << "runtime_inspection.batch_size=" << inspection->batch_size << '\n';
  std::cout << "runtime_inspection.safe_family_active="
            << inspection->safe_family_active << '\n';
  print_kernel_inspection("training_step", inspection->training_step);
  print_kernel_inspection("evaluate", inspection->evaluate);
  print_kernel_inspection("evaluate_with_gradient",
                          inspection->evaluate_with_gradient);
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    auto ctx = tmnn::MetalContext::create();
    if (!ctx || !ctx->is_gpu_available()) {
      return fail(ExitCode::NoGpu, "startup",
                  "mlp_learning_an_image requires a Metal-capable GPU");
    }

    tmnn::json config = default_config();
    if (argc >= 2) {
      std::cout << "Loading custom json config '" << argv[1] << "'.\n";
      std::ifstream f{argv[1]};
      if (!f) {
        return fail(ExitCode::ConfigOpenFailed, "config",
                    "failed to open JSON config file '" + std::string(argv[1]) +
                        "'");
      }
      try {
        config = tmnn::json::parse(f, nullptr, true, true);
      } catch (const std::exception &e) {
        return fail(ExitCode::ConfigParseFailed, "config",
                    std::string("failed to parse JSON config: ") + e.what());
      }
    }

    std::vector<float> image;
    make_image(image);
    std::vector<float> xs_and_ys;
    make_coords(xs_and_ys);

    if (!save_image(image, "reference.ppm")) {
      return fail(ExitCode::ReferenceWriteFailed, "write-reference",
                  "failed to write reference.ppm");
    }

    const uint32_t batch_size = kPixelCount;
    const uint32_t n_training_steps =
        argc >= 3 ? static_cast<uint32_t>(std::stoi(argv[2])) : 32u;
    const uint32_t n_input_dims = 3;
    const uint32_t n_output_dims = 3;

    auto trainer_result = tmnn::try_create_from_config(
        n_input_dims, n_output_dims, config, {.batch_size = batch_size}, ctx);
    if (!trainer_result) {
      if (!trainer_result.error().details.empty()) {
        print_config_diagnostics(trainer_result.error().details);
      }
      return fail(ExitCode::TrainerCreateFailed, "create-from-config",
                  "failed to construct trainer from config",
                  trainer_result.error());
    }
    auto trainer = std::move(*trainer_result);

    std::vector<float> training_batch_inputs = xs_and_ys;
    std::vector<float> training_batch_targets = image;
    std::vector<float> prediction(static_cast<size_t>(kPixelCount) * 3u);

    auto begin = std::chrono::steady_clock::now();
    float tmp_loss = 0.0f;
    uint32_t tmp_loss_counter = 0;

    std::cout << "Beginning optimization with " << n_training_steps
              << " training steps.\n";

    uint32_t interval = 4u;
    for (uint32_t i = 0; i < n_training_steps; ++i) {
      const bool print_loss = i % interval == 0;
      const auto result = trainer.training_step(training_batch_inputs.data(),
                                                training_batch_targets.data(),
                                                static_cast<int>(batch_size));
      if (!std::isfinite(result.loss)) {
        print_numerics(result);
        return fail(ExitCode::TrainingNonFinite, "training",
                    "training produced a non-finite loss");
      }

      tmp_loss += result.loss;
      ++tmp_loss_counter;

      if (print_loss) {
        const auto end = std::chrono::steady_clock::now();
        const auto micros =
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
        std::cout << "Step#" << i << ": loss="
                  << (tmp_loss / static_cast<float>(tmp_loss_counter))
                  << " time=" << micros << "[us]\n";
        tmp_loss = 0.0f;
        tmp_loss_counter = 0;
        begin = std::chrono::steady_clock::now();
        if (i > 0 && interval < 16u) {
          interval *= 2u;
        }
      }
    }

    if (!trainer.inference(xs_and_ys.data(), prediction.data(),
                           static_cast<int>(batch_size))) {
      return fail(ExitCode::InferenceFailed, "inference",
                  "trainer.inference(...) returned false",
                  trainer.last_diagnostic());
    }

    if (runtime_inspection_enabled()) {
      print_runtime_inspection(trainer);
    }

    const std::string output_filename = argc >= 4 ? argv[3] : "learned.ppm";
    if (!save_image(prediction, output_filename)) {
      return fail(ExitCode::OutputWriteFailed, "write-output",
                  "failed to write '" + output_filename + "'");
    }

    std::cout << "final_loss=" << loss(prediction, image) << '\n';
    std::cout << "Wrote reference.ppm and " << output_filename << '\n';
    std::cout << "exit_code=0\n";
    return static_cast<int>(ExitCode::Success);
  } catch (const std::exception &e) {
    return fail(ExitCode::UncaughtException, "exception",
                std::string("uncaught exception: ") + e.what());
  }
}
