/**
 * @file autotune_manifest.cpp
 * @brief JSON persistence for tmnn autotune/prewarm manifests.
 */

#include "tiny-metal-nn/autotune_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace tmnn {
namespace {

using nlohmann::json;

std::string family_to_json(NetworkFamily family) {
  return std::string(to_string(family));
}

std::string reason_to_json(PlannerFallbackReason reason) {
  return std::string(to_string(reason));
}

std::string objective_to_json(AutotuneSearchObjective objective) {
  return std::string(to_string(objective));
}

json measurement_to_json(const AutotuneCandidateMeasurement &measurement) {
  json j;
  j["family"] = family_to_json(measurement.family);
  j["warmup_step_ns"] = measurement.warmup_step_ns;
  j["measured_step_ns"] = measurement.measured_step_ns;
  j["succeeded"] = measurement.succeeded;
  j["build_ns"] = measurement.build_ns;
  j["objective_ns"] = measurement.objective_ns;
  if (!measurement.failure.empty())
    j["failure"] = measurement.failure;
  return j;
}

NetworkFamily family_from_json(const json &j) {
  const auto value = j.get<std::string>();
  if (value == "FullyFusedMetal")
    return NetworkFamily::FullyFusedMetal;
  if (value == "TiledMetal")
    return NetworkFamily::TiledMetal;
  if (value == "SafeDebugMetal")
    return NetworkFamily::SafeDebugMetal;
  throw std::invalid_argument("load_autotune_manifest: unknown family '" +
                              value + "'");
}

PlannerFallbackReason reason_from_json(const json &j) {
  const auto value = j.get<std::string>();
  if (value == "None")
    return PlannerFallbackReason::None;
  if (value == "NoGPUContext")
    return PlannerFallbackReason::NoGPUContext;
  if (value == "UnsupportedBackendPreference")
    return PlannerFallbackReason::UnsupportedBackendPreference;
  if (value == "MissingSIMDGroupMatrix")
    return PlannerFallbackReason::MissingSIMDGroupMatrix;
  if (value == "ThreadgroupMemoryExceeded")
    return PlannerFallbackReason::ThreadgroupMemoryExceeded;
  if (value == "UnsupportedEncoding")
    return PlannerFallbackReason::UnsupportedEncoding;
  if (value == "UnsupportedInputDims")
    return PlannerFallbackReason::UnsupportedInputDims;
  if (value == "UnsupportedOutputDims")
    return PlannerFallbackReason::UnsupportedOutputDims;
  if (value == "MultiOutputFusedUnsupported")
    return PlannerFallbackReason::MultiOutputFusedUnsupported;
  if (value == "HiddenDimUnsupported")
    return PlannerFallbackReason::HiddenDimUnsupported;
  if (value == "ForcedTiled")
    return PlannerFallbackReason::ForcedTiled;
  if (value == "ForcedSafeDebug")
    return PlannerFallbackReason::ForcedSafeDebug;
  if (value == "EncodingRequiresTiledFamily")
    return PlannerFallbackReason::EncodingRequiresTiledFamily;
  throw std::invalid_argument("load_autotune_manifest: unknown planner reason '" +
                              value + "'");
}

AutotuneSearchObjective objective_from_json(const json &j) {
  const auto value = j.get<std::string>();
  if (value == "SteadyStateStep")
    return AutotuneSearchObjective::SteadyStateStep;
  if (value == "BuildPlusMeasureWindow")
    return AutotuneSearchObjective::BuildPlusMeasureWindow;
  throw std::invalid_argument(
      "load_autotune_manifest: unknown autotune objective '" + value + "'");
}

AutotuneCandidateMeasurement measurement_from_json(const json &j) {
  AutotuneCandidateMeasurement measurement;
  measurement.family = family_from_json(j.at("family"));
  measurement.warmup_step_ns = j.value("warmup_step_ns", 0ull);
  measurement.measured_step_ns = j.value("measured_step_ns", 0ull);
  measurement.succeeded = j.value("succeeded", false);
  measurement.failure = j.value("failure", std::string{});
  measurement.build_ns = j.value("build_ns", 0ull);
  measurement.objective_ns =
      j.value("objective_ns", measurement.measured_step_ns);
  return measurement;
}

json entry_to_json(const AutotuneManifestEntry &entry) {
  json j;
  j["planner_fingerprint"] = entry.planner_fingerprint;
  j["selected_family"] = family_to_json(entry.selected_family);
  j["family_was_forced"] = entry.family_was_forced;
  j["selected_by_measurement"] = entry.selected_by_measurement;
  j["measurement_batch_size"] = entry.measurement_batch_size;
  j["measurement_warmup_steps"] = entry.measurement_warmup_steps;
  j["measurement_steps"] = entry.measurement_steps;
  j["measurement_objective"] = objective_to_json(entry.measurement_objective);
  j["selected_family_measured_step_ns"] =
      entry.selected_family_measured_step_ns;
  j["selected_family_objective_ns"] = entry.selected_family_objective_ns;
  j["candidate_families"] = json::array();
  for (const auto family : entry.candidate_families)
    j["candidate_families"].push_back(family_to_json(family));
  j["reasons"] = json::array();
  for (const auto reason : entry.reasons)
    j["reasons"].push_back(reason_to_json(reason));
  j["measurements"] = json::array();
  for (const auto &measurement : entry.measurements)
    j["measurements"].push_back(measurement_to_json(measurement));
  return j;
}

AutotuneManifestEntry entry_from_json(const json &j) {
  AutotuneManifestEntry entry;
  entry.planner_fingerprint = j.at("planner_fingerprint").get<uint64_t>();
  entry.selected_family = family_from_json(j.at("selected_family"));
  entry.family_was_forced = j.value("family_was_forced", false);
  entry.selected_by_measurement = j.value("selected_by_measurement", false);
  entry.measurement_batch_size = j.value("measurement_batch_size", 0u);
  entry.measurement_warmup_steps = j.value("measurement_warmup_steps", 2u);
  entry.measurement_steps = j.value("measurement_steps", 0u);
  entry.measurement_objective =
      j.contains("measurement_objective")
          ? objective_from_json(j.at("measurement_objective"))
          : AutotuneSearchObjective::SteadyStateStep;
  entry.selected_family_measured_step_ns =
      j.value("selected_family_measured_step_ns", 0ull);
  entry.selected_family_objective_ns =
      j.value("selected_family_objective_ns",
              entry.selected_family_measured_step_ns);
  if (j.contains("candidate_families")) {
    for (const auto &family : j.at("candidate_families"))
      entry.candidate_families.push_back(family_from_json(family));
  }
  if (j.contains("reasons")) {
    for (const auto &reason : j.at("reasons"))
      entry.reasons.push_back(reason_from_json(reason));
  }
  if (j.contains("measurements")) {
    for (const auto &measurement : j.at("measurements"))
      entry.measurements.push_back(measurement_from_json(measurement));
  }
  return entry;
}

} // namespace

AutotuneManifest load_autotune_manifest(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  json j;
  in >> j;

  AutotuneManifest manifest;
  manifest.version = j.value("version", "");
  if (manifest.version != kAutotuneManifestVersion) {
    throw std::invalid_argument(
        "load_autotune_manifest: unsupported manifest version '" +
        manifest.version + "'");
  }

  if (j.contains("device")) {
    const auto &device = j.at("device");
    manifest.device_name = device.value("name", "");
    manifest.device_family = device.value("gpu_family", 0u);
  } else {
    manifest.device_name = j.value("device_name", "");
    manifest.device_family = j.value("device_family", 0u);
  }

  if (j.contains("entries")) {
    for (const auto &entry : j.at("entries"))
      manifest.entries.push_back(entry_from_json(entry));
  }

  std::sort(manifest.entries.begin(), manifest.entries.end(),
            [](const auto &a, const auto &b) {
              return a.planner_fingerprint < b.planner_fingerprint;
            });
  return manifest;
}

void save_autotune_manifest(const std::string &path,
                            const AutotuneManifest &manifest) {
  json j;
  j["version"] = manifest.version.empty() ? std::string(kAutotuneManifestVersion)
                                          : manifest.version;
  j["device"] = {{"name", manifest.device_name},
                 {"gpu_family", manifest.device_family}};
  j["entries"] = json::array();

  std::vector<AutotuneManifestEntry> entries = manifest.entries;
  std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
    return a.planner_fingerprint < b.planner_fingerprint;
  });
  for (const auto &entry : entries)
    j["entries"].push_back(entry_to_json(entry));

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("save_autotune_manifest: failed to open '" + path +
                             "' for writing");
  }
  out << j.dump(2) << '\n';
}

} // namespace tmnn
