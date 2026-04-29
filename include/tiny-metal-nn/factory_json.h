#pragma once

/**
 * @file factory_json.h
 * @brief Generic JSON-driven factory helpers for tmnn public descriptors.
 */

#include "tiny-metal-nn/default_trainer.h"
#include "tiny-metal-nn/detail/factory_json_detail.h"
#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/rotated_hash_grid.h"

#include <memory>
#include <utility>

namespace tmnn {

[[nodiscard]] inline CanonicalModelConfig
canonicalize_model_config(uint32_t n_input_dims, uint32_t n_output_dims,
                          const json &config) {
  CanonicalModelConfig canonical;
  const json encoding_config =
      config.contains("encoding") ? config.at("encoding") : json::object();
  const json network_config =
      config.contains("network") ? config.at("network") : json::object();
  canonical.encoding = detail::canonicalize_encoding_config(
      n_input_dims, encoding_config, canonical.diagnostics);
  const auto encoded_dims =
      canonical.encoding.at("n_levels").get<uint32_t>() *
      canonical.encoding.at("n_features_per_level").get<uint32_t>();
  canonical.network = detail::canonicalize_network_config(
      encoded_dims, n_output_dims, network_config, canonical.diagnostics);
  return canonical;
}

namespace detail {

[[nodiscard]] inline std::shared_ptr<Encoding>
build_encoding_from_canonical(const json &encoding_config) {
  const std::string otype = encoding_config.at("otype").get<std::string>();
  if (otype == "HashGrid") {
    HashGridEncoding::Config cfg{
        .num_levels = static_cast<int>(encoding_config.at("n_levels").get<uint32_t>()),
        .features_per_level = static_cast<int>(
            encoding_config.at("n_features_per_level").get<uint32_t>()),
        .log2_hashmap_size = static_cast<int>(
            encoding_config.at("log2_hashmap_size").get<uint32_t>()),
        .base_resolution =
            encoding_config.at("base_resolution").get<float>(),
        .per_level_scale =
            encoding_config.at("per_level_scale").get<float>(),
        .input_dims = static_cast<int>(
            encoding_config.at("n_input_dims").get<uint32_t>()),
    };
    return create_encoding(cfg);
  }
  if (otype == "RotatedMHE") {
    RotatedHashGridEncoding::Config cfg{
        .num_levels = static_cast<int>(encoding_config.at("n_levels").get<uint32_t>()),
        .features_per_level = static_cast<int>(
            encoding_config.at("n_features_per_level").get<uint32_t>()),
        .log2_hashmap_size = static_cast<int>(
            encoding_config.at("log2_hashmap_size").get<uint32_t>()),
        .base_resolution =
            encoding_config.at("base_resolution").get<float>(),
        .per_level_scale =
            encoding_config.at("per_level_scale").get<float>(),
        .input_dims = static_cast<int>(
            encoding_config.at("n_input_dims").get<uint32_t>()),
    };
    return create_rotated_encoding(cfg);
  }
  throw std::invalid_argument("unsupported canonical encoding otype '" + otype +
                              "'");
}

[[nodiscard]] inline std::shared_ptr<Network>
build_network_from_canonical(const json &network_config) {
  const std::string otype = network_config.at("otype").get<std::string>();
  if (otype != "FullyFusedMLP") {
    throw std::invalid_argument("unsupported canonical network otype '" + otype +
                                "'");
  }
  FullyFusedMLP::Config cfg{
      .hidden_dim =
          static_cast<int>(network_config.at("n_neurons").get<uint32_t>()),
      .num_hidden_layers = static_cast<int>(
          network_config.at("n_hidden_layers").get<uint32_t>()),
      .n_input =
          static_cast<int>(network_config.at("n_input_dims").get<uint32_t>()),
      .n_output =
          static_cast<int>(network_config.at("n_output_dims").get<uint32_t>()),
  };
  return create_network(cfg);
}

[[nodiscard]] inline std::shared_ptr<Loss>
build_loss_from_canonical(const json &loss_config) {
  const std::string otype = loss_config.at("otype").get<std::string>();
  if (otype == "L2")
    return create_loss_l2();
  if (otype == "L1")
    return create_loss_l1();
  if (otype == "Huber") {
    return create_loss_huber(loss_config.at("huber_delta").get<float>());
  }
  if (otype == "Cosine") {
    const auto output_dims = loss_config.contains("output_dims")
                                 ? loss_config.at("output_dims").get<uint32_t>()
                                 : 0u;
    return create_loss_cosine(output_dims);
  }
  throw std::invalid_argument("unsupported canonical loss otype '" + otype +
                              "'");
}

[[nodiscard]] inline std::shared_ptr<Optimizer>
build_optimizer_from_canonical(const json &optimizer_config) {
  const std::string otype = optimizer_config.at("otype").get<std::string>();
  if (otype != "Adam") {
    throw std::invalid_argument("unsupported canonical optimizer otype '" +
                                otype + "'");
  }
  Adam::Config cfg;
  cfg.lr = optimizer_config.at("learning_rate").get<float>();
  cfg.beta1 = optimizer_config.at("beta1").get<float>();
  cfg.beta2 = optimizer_config.at("beta2").get<float>();
  cfg.epsilon = optimizer_config.at("epsilon").get<float>();
  return create_optimizer_adam(cfg);
}

template <typename T>
[[nodiscard]] inline Result<T>
unexpected_from_exception(std::string_view surface, const std::exception &e) {
  return make_error_result<T>(diagnostic_from_exception(surface, e));
}

inline void emit_non_error_config_diagnostics(
    std::string_view surface, const std::vector<ConfigDiagnostic> &diagnostics) {
  if (!has_logger_hook())
    return;

  std::vector<DiagnosticDetail> details;
  for (const auto &diag : diagnostics) {
    if (!diag.is_error()) {
      details.push_back(diag);
    }
  }
  if (details.empty())
    return;

  emit_diagnostic(DiagnosticInfo{
      .code = DiagnosticCode::None,
      .operation = std::string(surface),
      .message = "configuration canonicalization notes",
      .details = std::move(details),
  });
}

} // namespace detail

[[nodiscard]] inline Result<std::shared_ptr<Encoding>>
try_create_encoding_from_json(uint32_t n_input_dims, const json &config) {
  std::vector<ConfigDiagnostic> diagnostics;
  auto canonical =
      detail::canonicalize_encoding_config(n_input_dims, config, diagnostics);
  if (auto diagnostic =
          detail::diagnostic_from_config_errors("create_encoding_from_json",
                                                diagnostics)) {
    return make_error_result<std::shared_ptr<Encoding>>(std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics("create_encoding_from_json",
                                            diagnostics);
  try {
    return detail::build_encoding_from_canonical(canonical);
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<std::shared_ptr<Encoding>>(
        "create_encoding_from_json", e);
  }
}

[[nodiscard]] inline Result<std::shared_ptr<Network>>
try_create_network_from_json(uint32_t n_input_dims, uint32_t n_output_dims,
                             const json &config) {
  std::vector<ConfigDiagnostic> diagnostics;
  auto canonical = detail::canonicalize_network_config(
      n_input_dims, n_output_dims, config, diagnostics);
  if (auto diagnostic =
          detail::diagnostic_from_config_errors("create_network_from_json",
                                                diagnostics)) {
    return make_error_result<std::shared_ptr<Network>>(std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics("create_network_from_json",
                                            diagnostics);
  try {
    return detail::build_network_from_canonical(canonical);
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<std::shared_ptr<Network>>(
        "create_network_from_json", e);
  }
}

[[nodiscard]] inline Result<std::shared_ptr<NetworkWithInputEncoding>>
try_create_network_with_input_encoding_from_json(uint32_t n_input_dims,
                                                 uint32_t n_output_dims,
                                                 const json &config) {
  auto canonical = canonicalize_model_config(n_input_dims, n_output_dims, config);
  if (auto diagnostic = detail::diagnostic_from_config_errors(
          "create_network_with_input_encoding_from_json",
          canonical.diagnostics)) {
    return make_error_result<std::shared_ptr<NetworkWithInputEncoding>>(
        std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics(
      "create_network_with_input_encoding_from_json", canonical.diagnostics);
  try {
    return create_network_with_input_encoding(
        detail::build_encoding_from_canonical(canonical.encoding),
        detail::build_network_from_canonical(canonical.network));
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<
        std::shared_ptr<NetworkWithInputEncoding>>(
        "create_network_with_input_encoding_from_json", e);
  }
}

[[nodiscard]] inline Result<std::shared_ptr<Encoding>>
try_create_encoding(uint32_t n_input_dims, const json &config) {
  return try_create_encoding_from_json(n_input_dims, config);
}

[[nodiscard]] inline Result<std::shared_ptr<Network>>
try_create_network(uint32_t n_input_dims, uint32_t n_output_dims,
                   const json &config) {
  return try_create_network_from_json(n_input_dims, n_output_dims, config);
}

[[nodiscard]] inline Result<std::shared_ptr<NetworkWithInputEncoding>>
try_create_network_with_input_encoding(uint32_t n_input_dims,
                                       uint32_t n_output_dims,
                                       const json &config) {
  return try_create_network_with_input_encoding_from_json(n_input_dims,
                                                          n_output_dims,
                                                          config);
}

[[nodiscard]] inline Result<std::shared_ptr<Loss>>
try_create_loss(const json &config) {
  std::vector<ConfigDiagnostic> diagnostics;
  auto canonical = detail::canonicalize_loss_config(config, diagnostics);
  if (auto diagnostic =
          detail::diagnostic_from_config_errors("create_loss", diagnostics)) {
    return make_error_result<std::shared_ptr<Loss>>(std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics("create_loss", diagnostics);
  try {
    return detail::build_loss_from_canonical(canonical);
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<std::shared_ptr<Loss>>(
        "create_loss", e);
  }
}

[[nodiscard]] inline Result<std::shared_ptr<Optimizer>>
try_create_optimizer(const json &config) {
  std::vector<ConfigDiagnostic> diagnostics;
  auto canonical = detail::canonicalize_optimizer_config(config, diagnostics);
  if (auto diagnostic = detail::diagnostic_from_config_errors("create_optimizer",
                                                              diagnostics)) {
    return make_error_result<std::shared_ptr<Optimizer>>(std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics("create_optimizer", diagnostics);
  try {
    return detail::build_optimizer_from_canonical(canonical);
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<std::shared_ptr<Optimizer>>(
        "create_optimizer", e);
  }
}

[[nodiscard]] inline Result<Trainer>
try_create_from_config(uint32_t n_input_dims, uint32_t n_output_dims,
                       const json &config,
                       const TrainerConfig &trainer_cfg = default_trainer_config(),
                       std::shared_ptr<MetalContext> ctx = nullptr) {
  if (!config.is_object()) {
    return make_error_result<Trainer>(DiagnosticInfo{
        .code = DiagnosticCode::InvalidArgument,
        .operation = "create_from_config",
        .message = "config must be a JSON object",
        .details = {{DiagnosticSeverity::Error, "config",
                     "must be a JSON object"}},
    });
  }

  std::vector<ConfigDiagnostic> diagnostics;
  detail::validate_known_keys(config,
                              {"encoding", "network", "loss", "optimizer",
                               "weight_init", "batch_size", "training"},
                              "config", diagnostics);

  const auto canonical_model =
      canonicalize_model_config(n_input_dims, n_output_dims, config);
  diagnostics.insert(diagnostics.end(), canonical_model.diagnostics.begin(),
                     canonical_model.diagnostics.end());

  const auto canonical_loss = detail::canonicalize_loss_config(
      config.contains("loss") ? config.at("loss") : json{{"otype", "L2"}},
      diagnostics);
  const auto canonical_optimizer = detail::canonicalize_optimizer_config(
      config.contains("optimizer") ? config.at("optimizer")
                                   : json{{"otype", "Adam"}},
      diagnostics);
  const auto canonical_weight_init = detail::canonicalize_weight_init_config(
      config.contains("weight_init") ? config.at("weight_init")
                                     : json::object(),
      diagnostics);

  // Top-level batch_size (011 §1, ratified 2026-04-28). Optional; uses
  // trainer_cfg.batch_size as fallback when absent.
  int batch_size_value = trainer_cfg.batch_size;
  const bool has_batch_size = config.contains("batch_size");
  if (has_batch_size) {
    batch_size_value =
        detail::json_get<int>(config, "batch_size", trainer_cfg.batch_size,
                              "batch_size", diagnostics);
    detail::validate_positive_int(batch_size_value, "batch_size", diagnostics);
  }

  if (auto diagnostic =
          detail::diagnostic_from_config_errors("create_from_config",
                                                diagnostics)) {
    return make_error_result<Trainer>(std::move(*diagnostic));
  }
  detail::emit_non_error_config_diagnostics("create_from_config", diagnostics);

  try {
    auto model = create_network_with_input_encoding(
        detail::build_encoding_from_canonical(canonical_model.encoding),
        detail::build_network_from_canonical(canonical_model.network));
    auto loss = detail::build_loss_from_canonical(canonical_loss);
    auto optimizer = detail::build_optimizer_from_canonical(canonical_optimizer);
    auto resolved_train_cfg = detail::trainer_config_from_weight_init_config(
        canonical_weight_init,
        detail::trainer_config_from_loss_config(
            canonical_loss,
            detail::trainer_config_from_optimizer_config(canonical_optimizer,
                                                         trainer_cfg)));
    if (has_batch_size) {
      resolved_train_cfg.batch_size = batch_size_value;
    }
    return try_create_trainer(std::move(model), std::move(loss),
                              std::move(optimizer), resolved_train_cfg,
                              std::move(ctx));
  } catch (const std::exception &e) {
    return detail::unexpected_from_exception<Trainer>("create_from_config", e);
  }
}

[[nodiscard]] inline std::shared_ptr<Encoding>
create_encoding_from_json(uint32_t n_input_dims, const json &config) {
  return unwrap_or_throw(
      try_create_encoding_from_json(n_input_dims, config),
      "create_encoding_from_json");
}

[[nodiscard]] inline std::shared_ptr<Network>
create_network_from_json(uint32_t n_input_dims, uint32_t n_output_dims,
                         const json &config) {
  return unwrap_or_throw(
      try_create_network_from_json(n_input_dims, n_output_dims, config),
      "create_network_from_json");
}

[[nodiscard]] inline std::shared_ptr<NetworkWithInputEncoding>
create_network_with_input_encoding_from_json(uint32_t n_input_dims,
                                             uint32_t n_output_dims,
                                             const json &config) {
  return unwrap_or_throw(try_create_network_with_input_encoding_from_json(
                             n_input_dims, n_output_dims, config),
                         "create_network_with_input_encoding_from_json");
}

[[nodiscard]] inline std::shared_ptr<Encoding>
create_encoding(uint32_t n_input_dims, const json &config) {
  return create_encoding_from_json(n_input_dims, config);
}

[[nodiscard]] inline std::shared_ptr<Network>
create_network(uint32_t n_input_dims, uint32_t n_output_dims,
               const json &config) {
  return create_network_from_json(n_input_dims, n_output_dims, config);
}

[[nodiscard]] inline std::shared_ptr<NetworkWithInputEncoding>
create_network_with_input_encoding(uint32_t n_input_dims,
                                   uint32_t n_output_dims,
                                   const json &config) {
  return create_network_with_input_encoding_from_json(n_input_dims,
                                                      n_output_dims, config);
}

[[nodiscard]] inline std::shared_ptr<Loss> create_loss(const json &config) {
  return unwrap_or_throw(try_create_loss(config), "create_loss");
}

[[nodiscard]] inline std::shared_ptr<Optimizer>
create_optimizer(const json &config) {
  return unwrap_or_throw(try_create_optimizer(config), "create_optimizer");
}

[[nodiscard]] inline Trainer
create_from_config(uint32_t n_input_dims, uint32_t n_output_dims,
                   const json &config,
                   const TrainerConfig &trainer_cfg = default_trainer_config(),
                   std::shared_ptr<MetalContext> ctx = nullptr) {
  return unwrap_or_throw(try_create_from_config(n_input_dims, n_output_dims,
                                                config, trainer_cfg,
                                                std::move(ctx)),
                         "create_from_config");
}

} // namespace tmnn
