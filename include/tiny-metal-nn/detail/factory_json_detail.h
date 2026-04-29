#pragma once

/**
 * @file detail/factory_json_detail.h
 * @brief Canonicalization and diagnostics helpers shared by factory_json.h.
 *
 * Internal/public-support header: include `tiny-metal-nn/factory_json.h`
 * from user code unless you explicitly need the lower-level helpers.
 */

#include "tiny-metal-nn/trainer.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tmnn {

using json = nlohmann::json;

struct CanonicalModelConfig {
  json encoding;
  json network;
  std::vector<ConfigDiagnostic> diagnostics;

  [[nodiscard]] bool has_errors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [](const ConfigDiagnostic &diag) {
                         return diag.severity == DiagnosticSeverity::Error;
                       });
  }
};

namespace detail {

inline void push_diagnostic(std::vector<ConfigDiagnostic> &diagnostics,
                            DiagnosticSeverity severity,
                            std::string_view path,
                            std::string_view message) {
  diagnostics.push_back(
      {severity, std::string(path), std::string(message)});
}

inline void validate_object(const json &config, std::string_view path,
                            std::vector<ConfigDiagnostic> &diagnostics) {
  if (!config.is_object()) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error, path,
                    "must be a JSON object");
  }
}

inline bool is_allowed_key(std::string_view key,
                           std::initializer_list<std::string_view> allowed) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

inline void validate_known_keys(
    const nlohmann::json &config,
    std::initializer_list<std::string_view> allowed,
    std::string_view prefix,
    std::vector<ConfigDiagnostic> &diagnostics) {
  for (const auto &item : config.items()) {
    if (!is_allowed_key(item.key(), allowed)) {
      push_diagnostic(
          diagnostics, DiagnosticSeverity::Error,
          std::string(prefix) + "." + item.key(),
          "unsupported field for current C4a config bridge");
    }
  }
}

template <typename T>
inline T json_get(const nlohmann::json &config, std::string_view key,
                  T default_value, std::string_view path,
                  std::vector<ConfigDiagnostic> &diagnostics) {
  if (!config.contains(std::string(key)))
    return default_value;

  try {
    return config.at(std::string(key)).get<T>();
  } catch (const nlohmann::json::exception &) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error, path,
                    "invalid JSON type");
    return default_value;
  }
}

inline void validate_positive_int(int value, std::string_view path,
                                  std::vector<ConfigDiagnostic> &diagnostics) {
  if (value <= 0) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error, path,
                    "must be positive");
  }
}

inline void validate_positive_float(float value, std::string_view path,
                                    std::vector<ConfigDiagnostic> &diagnostics) {
  if (value <= 0.0f) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error, path,
                    "must be positive");
  }
}

inline std::string normalize_encoding_otype(
    const json &config,
    std::vector<ConfigDiagnostic> &diagnostics) {
  const std::string raw =
      json_get<std::string>(config, "otype", "HashGrid", "encoding.otype",
                            diagnostics);
  if (raw == "HashGrid")
    return raw;
  if (raw == "MultiresolutionHashGrid") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Info,
                    "encoding.otype",
                    "normalized alias 'MultiresolutionHashGrid' to 'HashGrid'");
    return "HashGrid";
  }
  if (raw == "RotatedMHE")
    return raw;
  if (raw == "RotatedMultiresHashGrid") {
    push_diagnostic(
        diagnostics, DiagnosticSeverity::Info, "encoding.otype",
        "normalized alias 'RotatedMultiresHashGrid' to 'RotatedMHE'");
    return "RotatedMHE";
  }

  push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                  "encoding.otype",
                  "unsupported encoding otype '" + raw + "'");
  return raw;
}

inline std::string normalize_network_otype(
    const json &config,
    std::vector<ConfigDiagnostic> &diagnostics) {
  const std::string raw =
      json_get<std::string>(config, "otype", "FullyFusedMLP", "network.otype",
                            diagnostics);
  if (raw == "FullyFusedMLP")
    return raw;
  if (raw == "TiledMLP") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "network.otype",
                    "TiledMLP JSON canonicalization is reserved for later C4");
    return raw;
  }

  push_diagnostic(diagnostics, DiagnosticSeverity::Error, "network.otype",
                  "unsupported network otype '" + raw + "'");
  return raw;
}

inline json canonicalize_encoding_config(
    uint32_t n_input_dims, const json &config,
    std::vector<ConfigDiagnostic> &diagnostics) {
  validate_known_keys(config,
                      {"otype", "n_levels", "n_features_per_level",
                       "log2_hashmap_size", "base_resolution",
                       "per_level_scale", "interpolation"},
                      "encoding", diagnostics);

  if (n_input_dims == 0) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "encoding.n_input_dims", "must be >= 1");
  }

  const std::string otype = normalize_encoding_otype(config, diagnostics);
  const int n_levels = json_get<int>(config, "n_levels", 16, "encoding.n_levels",
                                     diagnostics);
  const int n_features_per_level =
      json_get<int>(config, "n_features_per_level", 2,
                    "encoding.n_features_per_level", diagnostics);
  const int log2_hashmap_size =
      json_get<int>(config, "log2_hashmap_size", 19,
                    "encoding.log2_hashmap_size", diagnostics);
  const float base_resolution =
      json_get<float>(config, "base_resolution", 16.0f,
                      "encoding.base_resolution", diagnostics);
  const float per_level_scale =
      json_get<float>(config, "per_level_scale", 1.447f,
                      "encoding.per_level_scale", diagnostics);
  const std::string interpolation =
      json_get<std::string>(config, "interpolation", "Linear",
                            "encoding.interpolation", diagnostics);

  validate_positive_int(n_levels, "encoding.n_levels", diagnostics);
  validate_positive_int(n_features_per_level,
                        "encoding.n_features_per_level", diagnostics);
  validate_positive_int(log2_hashmap_size, "encoding.log2_hashmap_size",
                        diagnostics);
  validate_positive_float(base_resolution, "encoding.base_resolution",
                          diagnostics);
  validate_positive_float(per_level_scale, "encoding.per_level_scale",
                          diagnostics);

  if (interpolation != "Linear") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "encoding.interpolation",
                    "only 'Linear' interpolation is supported by the current "
                    "core descriptor bridge");
  }

  return {
      {"otype", otype},
      {"n_input_dims", n_input_dims},
      {"n_levels", n_levels},
      {"n_features_per_level", n_features_per_level},
      {"log2_hashmap_size", log2_hashmap_size},
      {"base_resolution", base_resolution},
      {"per_level_scale", per_level_scale},
      {"interpolation", "Linear"},
  };
}

inline json canonicalize_network_config(
    uint32_t n_input_dims, uint32_t n_output_dims, const json &config,
    std::vector<ConfigDiagnostic> &diagnostics) {
  validate_known_keys(config,
                      {"otype", "n_neurons", "n_hidden_layers", "activation",
                       "output_activation", "training"},
                      "network", diagnostics);

  if (n_input_dims == 0) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "network.n_input_dims", "must be >= 1");
  }
  if (n_output_dims == 0) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "network.n_output_dims", "must be >= 1");
  }

  const std::string otype = normalize_network_otype(config, diagnostics);
  const int n_neurons =
      json_get<int>(config, "n_neurons", 64, "network.n_neurons", diagnostics);
  const int n_hidden_layers = json_get<int>(config, "n_hidden_layers", 2,
                                            "network.n_hidden_layers",
                                            diagnostics);
  const std::string activation =
      json_get<std::string>(config, "activation", "ReLU",
                            "network.activation", diagnostics);
  std::string output_activation =
      json_get<std::string>(config, "output_activation", "None",
                            "network.output_activation", diagnostics);

  validate_positive_int(n_neurons, "network.n_neurons", diagnostics);
  validate_positive_int(n_hidden_layers, "network.n_hidden_layers",
                        diagnostics);

  if (activation != "ReLU") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "network.activation",
                    "only 'ReLU' activation is supported by the current "
                    "FullyFusedMLP descriptor bridge");
  }

  if (output_activation == "Linear") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Info,
                    "network.output_activation",
                    "normalized 'Linear' output activation to canonical "
                    "'None'");
    output_activation = "None";
  } else if (output_activation != "None") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "network.output_activation",
                    "only 'None' output activation is supported by the current "
                    "FullyFusedMLP descriptor bridge");
  }

  if (config.contains("training")) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Info,
                    "network.training",
                    "ignored by descriptor-only JSON factories");
  }

  return {
      {"otype", otype},
      {"n_input_dims", n_input_dims},
      {"n_output_dims", n_output_dims},
      {"n_neurons", n_neurons},
      {"n_hidden_layers", n_hidden_layers},
      {"activation", "ReLU"},
      {"output_activation", output_activation},
  };
}

inline void throw_if_errors(std::string_view surface,
                            const std::vector<ConfigDiagnostic> &diagnostics) {
  std::ostringstream error_stream;
  bool has_errors = false;
  for (const auto &diag : diagnostics) {
    if (diag.severity != DiagnosticSeverity::Error)
      continue;
    if (has_errors)
      error_stream << "; ";
    error_stream << diag.path << ": " << diag.message;
    has_errors = true;
  }
  if (has_errors) {
    throw std::invalid_argument(std::string(surface) + ": " +
                                error_stream.str());
  }
}

[[nodiscard]] inline std::string
format_error_diagnostics(const std::vector<ConfigDiagnostic> &diagnostics) {
  std::ostringstream error_stream;
  bool has_errors = false;
  for (const auto &diag : diagnostics) {
    if (diag.severity != DiagnosticSeverity::Error)
      continue;
    if (has_errors)
      error_stream << "; ";
    error_stream << diag.path << ": " << diag.message;
    has_errors = true;
  }
  return error_stream.str();
}

[[nodiscard]] inline std::optional<DiagnosticInfo>
diagnostic_from_config_errors(std::string_view surface,
                              const std::vector<ConfigDiagnostic> &diagnostics) {
  const std::string message = format_error_diagnostics(diagnostics);
  if (message.empty())
    return std::nullopt;
  return DiagnosticInfo{
      .code = DiagnosticCode::InvalidArgument,
      .operation = std::string(surface),
      .message = std::move(message),
      .details = [&diagnostics]() {
        std::vector<DiagnosticDetail> details;
        for (const auto &diag : diagnostics) {
          if (diag.severity == DiagnosticSeverity::Error)
            details.push_back(diag);
        }
        return details;
      }(),
  };
}

[[nodiscard]] inline DiagnosticInfo
diagnostic_from_exception(std::string_view surface, const std::exception &e) {
  const auto *invalid_arg = dynamic_cast<const std::invalid_argument *>(&e);
  return DiagnosticInfo{
      .code = invalid_arg ? DiagnosticCode::InvalidArgument
                          : DiagnosticCode::OperationFailed,
      .operation = std::string(surface),
      .message = e.what(),
  };
}

inline void throw_from_diagnostic(const DiagnosticInfo &diagnostic) {
  const std::string message =
      diagnostic.operation.empty()
          ? diagnostic.message
          : (diagnostic.operation + ": " + diagnostic.message);
  if (diagnostic.code == DiagnosticCode::InvalidArgument) {
    throw std::invalid_argument(message);
  }
  throw std::runtime_error(message);
}

inline std::string normalize_loss_otype(
    const json &config, std::vector<ConfigDiagnostic> &diagnostics) {
  const std::string raw =
      json_get<std::string>(config, "otype", "L2", "loss.otype", diagnostics);
  if (raw == "L2" || raw == "L1" || raw == "Huber" || raw == "Cosine")
    return raw;
  push_diagnostic(diagnostics, DiagnosticSeverity::Error, "loss.otype",
                  "unsupported loss otype '" + raw + "'");
  return raw;
}

inline json canonicalize_loss_config(
    const json &config, std::vector<ConfigDiagnostic> &diagnostics) {
  validate_object(config, "loss", diagnostics);
  const std::string otype = normalize_loss_otype(config, diagnostics);
  if (otype == "Huber") {
    validate_known_keys(config, {"otype", "huber_delta"}, "loss", diagnostics);
    const float huber_delta = json_get<float>(config, "huber_delta", 1.0f,
                                              "loss.huber_delta", diagnostics);
    validate_positive_float(huber_delta, "loss.huber_delta", diagnostics);
    return {{"otype", otype}, {"huber_delta", huber_delta}};
  }
  if (otype == "Cosine") {
    validate_known_keys(config, {"otype", "output_dims"}, "loss", diagnostics);
    if (config.contains("output_dims")) {
      const int output_dims =
          json_get<int>(config, "output_dims", 0, "loss.output_dims",
                        diagnostics);
      if (output_dims < 2) {
        push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                        "loss.output_dims", "must be >= 2 when provided");
      }
      return {{"otype", otype}, {"output_dims", output_dims}};
    }
    return {{"otype", otype}};
  }
  validate_known_keys(config, {"otype"}, "loss", diagnostics);
  return {{"otype", otype}};
}

inline std::string normalize_optimizer_otype(
    const json &config, std::vector<ConfigDiagnostic> &diagnostics) {
  const std::string raw = json_get<std::string>(
      config, "otype", "Adam", "optimizer.otype", diagnostics);
  if (raw == "Adam")
    return raw;
  push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                  "optimizer.otype",
                  "unsupported optimizer otype '" + raw + "'");
  return raw;
}

inline json canonicalize_optimizer_config(
    const json &config, std::vector<ConfigDiagnostic> &diagnostics) {
  validate_object(config, "optimizer", diagnostics);
  validate_known_keys(config,
                      {"otype", "learning_rate", "beta1", "beta2", "epsilon",
                       "l1_reg", "l2_reg"},
                      "optimizer", diagnostics);
  const std::string otype = normalize_optimizer_otype(config, diagnostics);
  const float learning_rate = json_get<float>(
      config, "learning_rate", 1e-3f, "optimizer.learning_rate", diagnostics);
  const float beta1 =
      json_get<float>(config, "beta1", 0.9f, "optimizer.beta1", diagnostics);
  const float beta2 =
      json_get<float>(config, "beta2", 0.99f, "optimizer.beta2", diagnostics);
  const float epsilon = json_get<float>(config, "epsilon", 1e-15f,
                                        "optimizer.epsilon", diagnostics);
  const float l1_reg =
      json_get<float>(config, "l1_reg", 0.0f, "optimizer.l1_reg", diagnostics);
  const float l2_reg =
      json_get<float>(config, "l2_reg", 0.0f, "optimizer.l2_reg", diagnostics);

  validate_positive_float(learning_rate, "optimizer.learning_rate",
                          diagnostics);
  validate_positive_float(beta1, "optimizer.beta1", diagnostics);
  validate_positive_float(beta2, "optimizer.beta2", diagnostics);
  validate_positive_float(epsilon, "optimizer.epsilon", diagnostics);
  if (l1_reg < 0.0f) {
      push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "optimizer.l1_reg", "must be >= 0");
  }
  if (l2_reg < 0.0f) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "optimizer.l2_reg", "must be >= 0");
  }

  return {{"otype", otype},
          {"learning_rate", learning_rate},
          {"beta1", beta1},
          {"beta2", beta2},
          {"epsilon", epsilon},
          {"l1_reg", l1_reg},
          {"l2_reg", l2_reg}};
}

inline TrainerConfig trainer_config_from_optimizer_config(
    const json &config, const TrainerConfig &base = {}) {
  TrainerConfig out = base;
  out.lr_encoding = config.value("learning_rate", out.lr_encoding);
  out.lr_network = config.value("learning_rate", out.lr_network);
  out.beta1 = config.value("beta1", out.beta1);
  out.beta2 = config.value("beta2", out.beta2);
  out.epsilon = config.value("epsilon", out.epsilon);
  out.l1_reg = config.value("l1_reg", out.l1_reg);
  out.l2_reg = config.value("l2_reg", out.l2_reg);
  return out;
}

inline TrainerConfig trainer_config_from_loss_config(
    const json &config, const TrainerConfig &base = {}) {
  TrainerConfig out = base;
  const std::string otype = config.value("otype", "L2");
  if (otype == "L1") {
    out.loss_kind = extension::LossKind::L1;
    out.huber_delta = 1.0f;
    return out;
  }
  if (otype == "Huber") {
    out.loss_kind = extension::LossKind::Huber;
    out.huber_delta = config.value("huber_delta", out.huber_delta);
    return out;
  }
  if (otype == "Cosine") {
    out.loss_kind = extension::LossKind::Cosine;
    out.huber_delta = 1.0f;
    return out;
  }
  out.loss_kind = extension::LossKind::L2;
  out.huber_delta = 1.0f;
  return out;
}

// ── weight_init section (011 §6, ratified 2026-04-28) ────────────────────
// JSON-parseable WeightInitConfig. Default-constructed when "weight_init" is
// absent from the top-level config; per-field defaults match weight_init.h.

// Trusting parsers: assume the canonical string was already validated by the
// canonicalize pass, so these never push diagnostics.

inline HashGridInit hash_grid_init_from_string(const std::string &s) {
  if (s == "Zero")
    return HashGridInit::Zero;
  return HashGridInit::Uniform;
}

inline MlpInit mlp_init_from_string(const std::string &s) {
  if (s == "KaimingNormal")
    return MlpInit::KaimingNormal;
  if (s == "XavierUniform")
    return MlpInit::XavierUniform;
  if (s == "XavierNormal")
    return MlpInit::XavierNormal;
  if (s == "Uniform")
    return MlpInit::Uniform;
  if (s == "Normal")
    return MlpInit::Normal;
  if (s == "Zero")
    return MlpInit::Zero;
  return MlpInit::KaimingUniform;
}

inline MlpNonlinearity mlp_nonlinearity_from_string(const std::string &s) {
  if (s == "Linear")
    return MlpNonlinearity::Linear;
  if (s == "LeakyReLU")
    return MlpNonlinearity::LeakyReLU;
  if (s == "Tanh")
    return MlpNonlinearity::Tanh;
  if (s == "Sigmoid")
    return MlpNonlinearity::Sigmoid;
  return MlpNonlinearity::ReLU;
}

inline json canonicalize_weight_init_config(
    const json &config, std::vector<ConfigDiagnostic> &diagnostics) {
  validate_object(config, "weight_init", diagnostics);
  validate_known_keys(config,
                      {"hash_grid_init", "hash_grid_range", "mlp_init",
                       "mlp_nonlinearity", "mlp_uniform_range",
                       "mlp_normal_stddev", "mlp_kaiming_a", "seed"},
                      "weight_init", diagnostics);

  const std::string hash_grid_init = json_get<std::string>(
      config, "hash_grid_init", "Uniform", "weight_init.hash_grid_init",
      diagnostics);
  if (hash_grid_init != "Uniform" && hash_grid_init != "Zero") {
    push_diagnostic(
        diagnostics, DiagnosticSeverity::Error, "weight_init.hash_grid_init",
        "unsupported value '" + hash_grid_init +
            "'; allowed: Uniform, Zero");
  }

  const float hash_grid_range = json_get<float>(
      config, "hash_grid_range", 1.0e-4f, "weight_init.hash_grid_range",
      diagnostics);
  validate_positive_float(hash_grid_range, "weight_init.hash_grid_range",
                          diagnostics);

  const std::string mlp_init = json_get<std::string>(
      config, "mlp_init", "KaimingUniform", "weight_init.mlp_init",
      diagnostics);
  if (mlp_init != "KaimingUniform" && mlp_init != "KaimingNormal" &&
      mlp_init != "XavierUniform" && mlp_init != "XavierNormal" &&
      mlp_init != "Uniform" && mlp_init != "Normal" && mlp_init != "Zero") {
    push_diagnostic(
        diagnostics, DiagnosticSeverity::Error, "weight_init.mlp_init",
        "unsupported value '" + mlp_init +
            "'; allowed: KaimingUniform, KaimingNormal, XavierUniform, "
            "XavierNormal, Uniform, Normal, Zero");
  }

  const std::string mlp_nonlinearity = json_get<std::string>(
      config, "mlp_nonlinearity", "ReLU", "weight_init.mlp_nonlinearity",
      diagnostics);
  if (mlp_nonlinearity != "Linear" && mlp_nonlinearity != "ReLU" &&
      mlp_nonlinearity != "LeakyReLU" && mlp_nonlinearity != "Tanh" &&
      mlp_nonlinearity != "Sigmoid") {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "weight_init.mlp_nonlinearity",
                    "unsupported value '" + mlp_nonlinearity +
                        "'; allowed: Linear, ReLU, LeakyReLU, Tanh, Sigmoid");
  }

  const float mlp_uniform_range = json_get<float>(
      config, "mlp_uniform_range", 1.0e-2f, "weight_init.mlp_uniform_range",
      diagnostics);
  validate_positive_float(mlp_uniform_range, "weight_init.mlp_uniform_range",
                          diagnostics);

  const float mlp_normal_stddev = json_get<float>(
      config, "mlp_normal_stddev", 1.0e-2f, "weight_init.mlp_normal_stddev",
      diagnostics);
  validate_positive_float(mlp_normal_stddev, "weight_init.mlp_normal_stddev",
                          diagnostics);

  const float mlp_kaiming_a = json_get<float>(
      config, "mlp_kaiming_a", 0.0f, "weight_init.mlp_kaiming_a", diagnostics);
  if (mlp_kaiming_a < 0.0f) {
    push_diagnostic(diagnostics, DiagnosticSeverity::Error,
                    "weight_init.mlp_kaiming_a", "must be >= 0");
  }

  const uint64_t seed = json_get<uint64_t>(config, "seed", 42u,
                                            "weight_init.seed", diagnostics);

  return {
      {"hash_grid_init", hash_grid_init},
      {"hash_grid_range", hash_grid_range},
      {"mlp_init", mlp_init},
      {"mlp_nonlinearity", mlp_nonlinearity},
      {"mlp_uniform_range", mlp_uniform_range},
      {"mlp_normal_stddev", mlp_normal_stddev},
      {"mlp_kaiming_a", mlp_kaiming_a},
      {"seed", seed},
  };
}

inline TrainerConfig trainer_config_from_weight_init_config(
    const json &config, const TrainerConfig &base = {}) {
  TrainerConfig out = base;
  WeightInitConfig &wi = out.weight_init;
  wi.hash_grid_mode = hash_grid_init_from_string(
      config.value("hash_grid_init", std::string("Uniform")));
  wi.hash_grid_range = config.value("hash_grid_range", wi.hash_grid_range);
  wi.mlp_mode = mlp_init_from_string(
      config.value("mlp_init", std::string("KaimingUniform")));
  wi.mlp_nonlinearity = mlp_nonlinearity_from_string(
      config.value("mlp_nonlinearity", std::string("ReLU")));
  wi.mlp_uniform_range =
      config.value("mlp_uniform_range", wi.mlp_uniform_range);
  wi.mlp_normal_stddev =
      config.value("mlp_normal_stddev", wi.mlp_normal_stddev);
  wi.mlp_kaiming_a = config.value("mlp_kaiming_a", wi.mlp_kaiming_a);
  wi.seed = config.value("seed", wi.seed);
  return out;
}

} // namespace detail

} // namespace tmnn
