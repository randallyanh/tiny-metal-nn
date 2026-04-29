/**
 * @file test_factory_json.cpp
 * @brief Tests for the C4a JSON config bridge.
 */

#include "tiny-metal-nn/factory_json.h"

#include <gtest/gtest.h>

using namespace tmnn;
using json = nlohmann::json;

TEST(FactoryJson, CanonicalizesAliasesAndDefaults) {
  json config = {
      {"encoding",
       {{"otype", "MultiresolutionHashGrid"},
        {"n_levels", 12},
        {"n_features_per_level", 4}}},
      {"network",
       {{"n_neurons", 32},
        {"training", {{"batch_size", 8192}}},
        {"output_activation", "Linear"}}},
  };

  auto canonical = canonicalize_model_config(4, 2, config);

  EXPECT_FALSE(canonical.has_errors());
  EXPECT_EQ(canonical.encoding.at("otype"), "HashGrid");
  EXPECT_EQ(canonical.encoding.at("n_input_dims"), 4);
  EXPECT_EQ(canonical.network.at("otype"), "FullyFusedMLP");
  EXPECT_EQ(canonical.network.at("n_output_dims"), 2);
  EXPECT_EQ(canonical.network.at("output_activation"), "None");
  EXPECT_FALSE(canonical.diagnostics.empty());
}

TEST(FactoryJson, CreateEncodingFromJson) {
  json enc = {{"otype", "HashGrid"},
              {"n_levels", 8},
              {"n_features_per_level", 2},
              {"log2_hashmap_size", 17},
              {"base_resolution", 8.0f},
              {"per_level_scale", 1.5f}};

  auto encoding = create_encoding_from_json(4, enc);
  auto hash_grid = std::dynamic_pointer_cast<HashGridEncoding>(encoding);
  ASSERT_NE(hash_grid, nullptr);
  EXPECT_EQ(hash_grid->config().input_dims, 4);
  EXPECT_EQ(hash_grid->config().num_levels, 8);
  EXPECT_EQ(hash_grid->config().features_per_level, 2);
}

TEST(FactoryJson, CreateNetworkFromJson) {
  json net = {{"otype", "FullyFusedMLP"},
              {"n_neurons", 48},
              {"n_hidden_layers", 3},
              {"activation", "ReLU"},
              {"output_activation", "None"}};

  auto network = create_network_from_json(32, 4, net);
  auto mlp = std::dynamic_pointer_cast<FullyFusedMLP>(network);
  ASSERT_NE(mlp, nullptr);
  EXPECT_EQ(mlp->config().n_input, 32);
  EXPECT_EQ(mlp->config().n_output, 4);
  EXPECT_EQ(mlp->config().hidden_dim, 48);
  EXPECT_EQ(mlp->config().num_hidden_layers, 3);
}

TEST(FactoryJson, CreateNetworkWithInputEncodingFromJsonSupportsArbitraryDims) {
  json config = {
      {"encoding",
       {{"otype", "HashGrid"},
        {"n_levels", 10},
        {"n_features_per_level", 3},
        {"log2_hashmap_size", 17},
        {"base_resolution", 8.0f},
        {"per_level_scale", 1.6f}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"n_neurons", 32},
        {"n_hidden_layers", 2}}},
  };

  auto model = create_network_with_input_encoding_from_json(4, 3, config);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->n_input_dims(), 4);
  EXPECT_EQ(model->n_output_dims(), 3);
  EXPECT_EQ(model->encoding()->n_output_dims(), 30);
  EXPECT_EQ(model->network()->n_input_dims(), 30);
}

TEST(FactoryJson, RejectsUnsupportedInterpolation) {
  json config = {
      {"encoding", {{"otype", "HashGrid"}, {"interpolation", "Smoothstep"}}},
  };

  auto canonical = canonicalize_model_config(3, 1, config);
  EXPECT_TRUE(canonical.has_errors());
  EXPECT_THROW((void)create_encoding_from_json(3, config.at("encoding")),
               std::invalid_argument);
}

TEST(FactoryJson, RejectsUnsupportedActivation) {
  json net = {{"otype", "FullyFusedMLP"}, {"activation", "Sine"}};
  EXPECT_THROW((void)create_network_from_json(32, 1, net), std::invalid_argument);
}

TEST(FactoryJson, RotatedMHECanonicalizesAndMaterializesAsCoreEncoding) {
  json config = {
      {"encoding", {{"otype", "RotatedMultiresHashGrid"}}},
  };

  auto canonical = canonicalize_model_config(3, 1, config);
  EXPECT_FALSE(canonical.has_errors());
  EXPECT_EQ(canonical.encoding.at("otype"), "RotatedMHE");
  auto encoding = create_encoding_from_json(3, config.at("encoding"));
  auto rotated = std::dynamic_pointer_cast<RotatedHashGridEncoding>(encoding);
  ASSERT_NE(rotated, nullptr);
  EXPECT_EQ(rotated->config().input_dims, 3);
}

TEST(FactoryJson, CreateNetworkWithInputEncodingFromJsonSupportsRotatedMHE) {
  json config = {
      {"encoding",
       {{"otype", "RotatedMHE"},
        {"n_levels", 10},
        {"n_features_per_level", 2}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"n_neurons", 32},
        {"n_hidden_layers", 2}}},
  };

  auto model = create_network_with_input_encoding_from_json(3, 1, config);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->encoding()->name(), "RotatedHashGridEncoding");
  EXPECT_EQ(model->n_input_dims(), 3);
  EXPECT_EQ(model->n_output_dims(), 1);
}

TEST(FactoryJson, GenericCreateLossFromJson) {
  json loss_cfg = {{"otype", "L2"}};

  auto loss = create_loss(loss_cfg);
  auto l2 = std::dynamic_pointer_cast<L2Loss>(loss);
  ASSERT_NE(l2, nullptr);
  EXPECT_EQ(loss->name(), "L2");
}

TEST(FactoryJson, GenericCreateLossFromJsonSupportsL1HuberAndCosine) {
  auto l1 = create_loss(json{{"otype", "L1"}});
  auto l1_loss = std::dynamic_pointer_cast<L1Loss>(l1);
  ASSERT_NE(l1_loss, nullptr);
  EXPECT_EQ(l1->name(), "L1");

  auto huber = create_loss(json{{"otype", "Huber"}, {"huber_delta", 0.25f}});
  auto huber_loss = std::dynamic_pointer_cast<HuberLoss>(huber);
  ASSERT_NE(huber_loss, nullptr);
  EXPECT_EQ(huber->name(), "Huber");
  EXPECT_FLOAT_EQ(huber_loss->delta(), 0.25f);

  auto cosine = create_loss(json{{"otype", "Cosine"}, {"output_dims", 4}});
  auto cosine_loss = std::dynamic_pointer_cast<CosineLoss>(cosine);
  ASSERT_NE(cosine_loss, nullptr);
  EXPECT_EQ(cosine->name(), "Cosine");
  EXPECT_EQ(cosine_loss->output_dims(), 4u);
}

TEST(FactoryJson, RejectsInvalidHuberLossConfig) {
  EXPECT_THROW((void)create_loss(json{{"otype", "Huber"}, {"huber_delta", 0.0f}}),
               std::invalid_argument);
}

TEST(FactoryJson, RejectsInvalidCosineLossConfig) {
  EXPECT_THROW((void)create_loss(json{{"otype", "Cosine"}, {"output_dims", 1}}),
               std::invalid_argument);
}

TEST(FactoryJson, GenericCreateOptimizerFromJson) {
  json optimizer_cfg = {{"otype", "Adam"},
                        {"learning_rate", 2e-3f},
                        {"beta1", 0.8f},
                        {"beta2", 0.95f},
                        {"epsilon", 1e-12f}};

  auto optimizer = create_optimizer(optimizer_cfg);
  auto adam = std::dynamic_pointer_cast<Adam>(optimizer);
  ASSERT_NE(adam, nullptr);
  EXPECT_FLOAT_EQ(adam->config().lr, 2e-3f);
  EXPECT_FLOAT_EQ(adam->config().beta1, 0.8f);
  EXPECT_FLOAT_EQ(adam->config().beta2, 0.95f);
  EXPECT_FLOAT_EQ(adam->config().epsilon, 1e-12f);
}

TEST(FactoryJson, JsonOverloadsMirrorNamedFactories) {
  json enc = {{"otype", "HashGrid"},
              {"n_levels", 8},
              {"n_features_per_level", 2}};
  json net = {{"otype", "FullyFusedMLP"},
              {"n_neurons", 32},
              {"n_hidden_layers", 2}};
  json model_cfg = {{"encoding", enc}, {"network", net}};

  auto encoding = create_encoding(3, enc);
  auto network = create_network(16, 1, net);
  auto model = create_network_with_input_encoding(3, 1, model_cfg);

  ASSERT_NE(encoding, nullptr);
  ASSERT_NE(network, nullptr);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->n_input_dims(), 3);
  EXPECT_EQ(model->n_output_dims(), 1);
}

TEST(FactoryJson, FormatConfigDiagnosticIncludesSeverityPathAndMessage) {
  const ConfigDiagnostic diagnostic{
      .severity = ConfigDiagnosticSeverity::Error,
      .path = "encoding.interpolation",
      .message = "unsupported value",
  };
  EXPECT_EQ(format_config_diagnostic(diagnostic),
            "Error encoding.interpolation: unsupported value");
}

TEST(FactoryJson, TryCreateEncodingFromJsonReturnsStructuredDiagnostic) {
  json enc = {{"otype", "HashGrid"}, {"interpolation", "Smoothstep"}};

  auto result = try_create_encoding_from_json(3, enc);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, DiagnosticCode::InvalidArgument);
  EXPECT_EQ(result.error().operation, "create_encoding_from_json");
  EXPECT_NE(result.error().message.find("encoding.interpolation"),
            std::string::npos);
  ASSERT_FALSE(result.error().details.empty());
  EXPECT_EQ(result.error().details.front().severity, DiagnosticSeverity::Error);
}

TEST(FactoryJson, TryCreateFromConfigReturnsStructuredConfigDiagnostic) {
  json config = {
      {"loss", {{"otype", "L2"}}},
      {"optimizer", {{"otype", "Adam"}}},
      {"encoding", {{"otype", "HashGrid"}}},
      {"network", {{"otype", "FullyFusedMLP"}, {"activation", "Sine"}}},
  };

  auto result = try_create_from_config(3, 1, config);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, DiagnosticCode::InvalidArgument);
  EXPECT_EQ(result.error().operation, "create_from_config");
  EXPECT_NE(result.error().message.find("network.activation"),
            std::string::npos);
  ASSERT_FALSE(result.error().details.empty());
}

TEST(FactoryJson, TryCreateFromConfigSucceedsWithoutDiagnostic) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";

  json config = {
      {"loss", {{"otype", "L2"}}},
      {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-4f}}},
      {"encoding",
       {{"otype", "HashGrid"},
        {"n_levels", 8},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 17}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"n_neurons", 32},
        {"n_hidden_layers", 2}}},
  };

  auto result =
      try_create_from_config(3, 1, config, default_trainer_config(), ctx);
  ASSERT_TRUE(result.has_value());
  auto trainer = std::move(*result);
  EXPECT_TRUE(trainer.is_gpu_available());
}

// ── weight_init section (011 §6, ratified 2026-04-28) ────────────────────

TEST(FactoryJson, WeightInitDefaultsCanonicalize) {
  std::vector<ConfigDiagnostic> diagnostics;
  json canonical =
      detail::canonicalize_weight_init_config(json::object(), diagnostics);

  EXPECT_TRUE(diagnostics.empty());
  EXPECT_EQ(canonical.at("hash_grid_init"), "Uniform");
  EXPECT_FLOAT_EQ(canonical.at("hash_grid_range").get<float>(), 1.0e-4f);
  EXPECT_EQ(canonical.at("mlp_init"), "KaimingUniform");
  EXPECT_EQ(canonical.at("mlp_nonlinearity"), "ReLU");
  EXPECT_FLOAT_EQ(canonical.at("mlp_uniform_range").get<float>(), 1.0e-2f);
  EXPECT_FLOAT_EQ(canonical.at("mlp_normal_stddev").get<float>(), 1.0e-2f);
  EXPECT_FLOAT_EQ(canonical.at("mlp_kaiming_a").get<float>(), 0.0f);
  EXPECT_EQ(canonical.at("seed").get<uint64_t>(), 42u);
}

TEST(FactoryJson, WeightInitAllMlpInitValuesAccepted) {
  for (const std::string &v :
       {"KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal",
        "Uniform", "Normal", "Zero"}) {
    std::vector<ConfigDiagnostic> diagnostics;
    json canonical = detail::canonicalize_weight_init_config(
        json{{"mlp_init", v}}, diagnostics);
    EXPECT_TRUE(diagnostics.empty()) << "rejected '" << v << "'";
    EXPECT_EQ(canonical.at("mlp_init"), v);
  }
}

TEST(FactoryJson, WeightInitAllNonlinearityValuesAccepted) {
  for (const std::string &v :
       {"Linear", "ReLU", "LeakyReLU", "Tanh", "Sigmoid"}) {
    std::vector<ConfigDiagnostic> diagnostics;
    json canonical = detail::canonicalize_weight_init_config(
        json{{"mlp_nonlinearity", v}}, diagnostics);
    EXPECT_TRUE(diagnostics.empty()) << "rejected '" << v << "'";
    EXPECT_EQ(canonical.at("mlp_nonlinearity"), v);
  }
}

TEST(FactoryJson, WeightInitRejectsInvalidEnumValueWithSuggestion) {
  std::vector<ConfigDiagnostic> diagnostics;
  detail::canonicalize_weight_init_config(json{{"mlp_init", "Bogus"}},
                                          diagnostics);
  bool found = false;
  for (const auto &d : diagnostics) {
    if (d.path == "weight_init.mlp_init" &&
        d.severity == DiagnosticSeverity::Error) {
      EXPECT_NE(d.message.find("Bogus"), std::string::npos);
      EXPECT_NE(d.message.find("KaimingUniform"), std::string::npos);
      found = true;
    }
  }
  EXPECT_TRUE(found) << "expected error diagnostic on weight_init.mlp_init";
}

TEST(FactoryJson, WeightInitRejectsNegativeRange) {
  std::vector<ConfigDiagnostic> diagnostics;
  detail::canonicalize_weight_init_config(
      json{{"mlp_uniform_range", -0.01f}}, diagnostics);
  bool found = false;
  for (const auto &d : diagnostics) {
    if (d.path == "weight_init.mlp_uniform_range" &&
        d.severity == DiagnosticSeverity::Error) {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST(FactoryJson, WeightInitRejectsUnknownKey) {
  std::vector<ConfigDiagnostic> diagnostics;
  detail::canonicalize_weight_init_config(
      json{{"mystery_field", 1.0f}}, diagnostics);
  bool found = false;
  for (const auto &d : diagnostics) {
    if (d.path == "weight_init.mystery_field" &&
        d.severity == DiagnosticSeverity::Error) {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST(FactoryJson, TrainerConfigFromWeightInitPlumbsValues) {
  json canonical = {
      {"hash_grid_init", "Zero"},
      {"hash_grid_range", 5.0e-5f},
      {"mlp_init", "KaimingNormal"},
      {"mlp_nonlinearity", "LeakyReLU"},
      {"mlp_uniform_range", 2.0e-2f},
      {"mlp_normal_stddev", 3.0e-2f},
      {"mlp_kaiming_a", 0.1f},
      {"seed", uint64_t{12345}},
  };
  TrainerConfig out =
      detail::trainer_config_from_weight_init_config(canonical, {});

  EXPECT_EQ(out.weight_init.hash_grid_mode, HashGridInit::Zero);
  EXPECT_FLOAT_EQ(out.weight_init.hash_grid_range, 5.0e-5f);
  EXPECT_EQ(out.weight_init.mlp_mode, MlpInit::KaimingNormal);
  EXPECT_EQ(out.weight_init.mlp_nonlinearity, MlpNonlinearity::LeakyReLU);
  EXPECT_FLOAT_EQ(out.weight_init.mlp_uniform_range, 2.0e-2f);
  EXPECT_FLOAT_EQ(out.weight_init.mlp_normal_stddev, 3.0e-2f);
  EXPECT_FLOAT_EQ(out.weight_init.mlp_kaiming_a, 0.1f);
  EXPECT_EQ(out.weight_init.seed, uint64_t{12345});
}

// ── batch_size top-level (011 §1, ratified 2026-04-28) ───────────────────

TEST(FactoryJson, BatchSizeTopLevelAccepted) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";

  json config = {
      {"encoding",
       {{"otype", "HashGrid"}, {"n_levels", 8}, {"log2_hashmap_size", 17}}},
      {"network", {{"otype", "FullyFusedMLP"}, {"n_neurons", 32}}},
      {"batch_size", 512},
  };

  auto result =
      try_create_from_config(3, 1, config, default_trainer_config(), ctx);
  ASSERT_TRUE(result.has_value());
  auto trainer = std::move(*result);
  EXPECT_EQ(trainer.batch_plan().max_batch_size, 512u);
}

TEST(FactoryJson, BatchSizeRejectsZero) {
  json config = {
      {"encoding", {{"otype", "HashGrid"}}},
      {"network", {{"otype", "FullyFusedMLP"}}},
      {"batch_size", 0},
  };

  auto result = try_create_from_config(3, 1, config);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, DiagnosticCode::InvalidArgument);
  EXPECT_NE(result.error().message.find("batch_size"), std::string::npos);
}

// ── reference example from 011 §9 — full default JSON canonicalizes ──────

TEST(FactoryJson, ReferenceExampleCanonicalizesCleanly) {
  // Mirrors docs/know-how/011-json-schema-frozen.md §9: the full default
  // config with every key explicit. Drift between this test and 011 §9 is
  // a P0 doc/code mismatch.
  json config = {
      {"encoding",
       {{"otype", "HashGrid"},
        {"n_levels", 16},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 19},
        {"base_resolution", 16.0f},
        {"per_level_scale", 1.447f},
        {"interpolation", "Linear"}}},
      {"network",
       {{"otype", "FullyFusedMLP"},
        {"n_neurons", 64},
        {"n_hidden_layers", 2},
        {"activation", "ReLU"},
        {"output_activation", "None"}}},
      {"loss", {{"otype", "L2"}}},
      {"optimizer",
       {{"otype", "Adam"},
        {"learning_rate", 1e-3f},
        {"beta1", 0.9f},
        {"beta2", 0.99f},
        {"epsilon", 1e-15f},
        {"l1_reg", 0.0f},
        {"l2_reg", 0.0f}}},
      {"weight_init",
       {{"hash_grid_init", "Uniform"},
        {"hash_grid_range", 1.0e-4f},
        {"mlp_init", "KaimingUniform"},
        {"mlp_nonlinearity", "ReLU"},
        {"mlp_uniform_range", 1.0e-2f},
        {"mlp_normal_stddev", 1.0e-2f},
        {"mlp_kaiming_a", 0.0f},
        {"seed", uint64_t{42}}}},
      {"batch_size", 1024},
  };

  auto canonical_model = canonicalize_model_config(3, 1, config);
  EXPECT_FALSE(canonical_model.has_errors());

  std::vector<ConfigDiagnostic> diagnostics;
  detail::canonicalize_loss_config(config.at("loss"), diagnostics);
  detail::canonicalize_optimizer_config(config.at("optimizer"), diagnostics);
  detail::canonicalize_weight_init_config(config.at("weight_init"),
                                          diagnostics);
  bool any_error = false;
  for (const auto &d : diagnostics) {
    if (d.severity == DiagnosticSeverity::Error) {
      ADD_FAILURE() << "unexpected error: " << d.path << ": " << d.message;
      any_error = true;
    }
  }
  EXPECT_FALSE(any_error);
}
