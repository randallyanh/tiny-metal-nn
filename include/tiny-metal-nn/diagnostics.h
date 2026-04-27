#pragma once

/**
 * @file diagnostics.h
 * @brief Unified diagnostics, logger hook, and Result<T> contract.
 */

#include <expected>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <cstdint>
#include <utility>
#include <vector>

namespace tmnn {

enum class DiagnosticSeverity : uint8_t { Info, Warning, Error };

struct DiagnosticDetail {
  DiagnosticSeverity severity = DiagnosticSeverity::Info;
  std::string path;
  std::string message;

  [[nodiscard]] bool is_error() const {
    return severity == DiagnosticSeverity::Error;
  }
};

using ConfigDiagnosticSeverity = DiagnosticSeverity;
using ConfigDiagnostic = DiagnosticDetail;

/// Stable diagnostic codes for public non-throwing APIs.
enum class DiagnosticCode : uint32_t {
  None = 0,
  InvalidArgument,
  NullObject,
  MissingRuntime,
  MissingRuntimeAuthority,
  MissingRuntimeContext,
  MissingHostVisibleConfigWeights,
  MissingRuntimeBuffer,
  GpuUnavailable,
  BatchSubmissionUnavailable,
  KernelCompilationFailed,
  UnsupportedOperation,
  UnsupportedModelType,
  UnsupportedLossType,
  ConfigurationConflict,
  SchemaMismatch,
  OperationFailed,
};

/// Structured failure payload for public non-throwing APIs.
struct DiagnosticInfo {
  DiagnosticCode code = DiagnosticCode::None;
  std::string operation;
  std::string message;
  std::vector<DiagnosticDetail> details;

  [[nodiscard]] bool has_error() const {
    return code != DiagnosticCode::None;
  }
};

[[nodiscard]] inline const char *
diagnostic_severity_name(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Info:
    return "Info";
  case DiagnosticSeverity::Warning:
    return "Warning";
  case DiagnosticSeverity::Error:
    return "Error";
  }
  return "Unknown";
}

[[nodiscard]] inline std::string
format_diagnostic_detail(const DiagnosticDetail &detail) {
  std::string out = diagnostic_severity_name(detail.severity);
  if (!detail.path.empty()) {
    out += " ";
    out += detail.path;
  }
  if (!detail.message.empty()) {
    out += ": ";
    out += detail.message;
  }
  return out;
}

[[nodiscard]] inline std::string
format_config_diagnostic(const ConfigDiagnostic &diagnostic) {
  return format_diagnostic_detail(diagnostic);
}

[[nodiscard]] inline const char *diagnostic_code_name(DiagnosticCode code) {
  switch (code) {
  case DiagnosticCode::None:
    return "None";
  case DiagnosticCode::InvalidArgument:
    return "InvalidArgument";
  case DiagnosticCode::NullObject:
    return "NullObject";
  case DiagnosticCode::MissingRuntime:
    return "MissingRuntime";
  case DiagnosticCode::MissingRuntimeAuthority:
    return "MissingRuntimeAuthority";
  case DiagnosticCode::MissingRuntimeContext:
    return "MissingRuntimeContext";
  case DiagnosticCode::MissingHostVisibleConfigWeights:
    return "MissingHostVisibleConfigWeights";
  case DiagnosticCode::MissingRuntimeBuffer:
    return "MissingRuntimeBuffer";
  case DiagnosticCode::GpuUnavailable:
    return "GpuUnavailable";
  case DiagnosticCode::BatchSubmissionUnavailable:
    return "BatchSubmissionUnavailable";
  case DiagnosticCode::KernelCompilationFailed:
    return "KernelCompilationFailed";
  case DiagnosticCode::UnsupportedOperation:
    return "UnsupportedOperation";
  case DiagnosticCode::UnsupportedModelType:
    return "UnsupportedModelType";
  case DiagnosticCode::UnsupportedLossType:
    return "UnsupportedLossType";
  case DiagnosticCode::ConfigurationConflict:
    return "ConfigurationConflict";
  case DiagnosticCode::SchemaMismatch:
    return "SchemaMismatch";
  case DiagnosticCode::OperationFailed:
    return "OperationFailed";
  }
  return "UnknownDiagnosticCode";
}

[[nodiscard]] inline std::string format_diagnostic(const DiagnosticInfo &info) {
  std::string out = diagnostic_code_name(info.code);
  if (!info.operation.empty()) {
    out += " during ";
    out += info.operation;
  }
  if (!info.message.empty()) {
    out += ": ";
    out += info.message;
  }
  return out;
}

using LoggerHook = std::function<void(const DiagnosticInfo &)>;

namespace detail {

inline std::mutex &logger_hook_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline LoggerHook &logger_hook_storage() {
  static LoggerHook hook;
  return hook;
}

} // namespace detail

/// Install a process-wide diagnostic logger hook. The latest hook wins.
inline void set_logger_hook(LoggerHook hook) {
  std::scoped_lock lock(detail::logger_hook_mutex());
  detail::logger_hook_storage() = std::move(hook);
}

inline void clear_logger_hook() {
  std::scoped_lock lock(detail::logger_hook_mutex());
  detail::logger_hook_storage() = {};
}

[[nodiscard]] inline bool has_logger_hook() {
  std::scoped_lock lock(detail::logger_hook_mutex());
  return static_cast<bool>(detail::logger_hook_storage());
}

inline void emit_diagnostic(const DiagnosticInfo &info) {
  LoggerHook hook;
  {
    std::scoped_lock lock(detail::logger_hook_mutex());
    hook = detail::logger_hook_storage();
  }
  if (hook) {
    hook(info);
  }
}

template <typename T> using Result = std::expected<T, DiagnosticInfo>;

template <typename T>
[[nodiscard]] inline Result<T> make_result(T value) {
  return Result<T>(std::move(value));
}

template <typename T>
[[nodiscard]] inline Result<T> make_error_result(DiagnosticInfo diagnostic) {
  emit_diagnostic(diagnostic);
  return std::unexpected(std::move(diagnostic));
}

inline void throw_from_diagnostic(const DiagnosticInfo &diagnostic) {
  const std::string message =
      diagnostic.operation.empty()
          ? diagnostic.message
          : (diagnostic.operation + ": " + diagnostic.message);
  switch (diagnostic.code) {
  case DiagnosticCode::InvalidArgument:
  case DiagnosticCode::NullObject:
  case DiagnosticCode::UnsupportedOperation:
  case DiagnosticCode::UnsupportedModelType:
  case DiagnosticCode::UnsupportedLossType:
  case DiagnosticCode::ConfigurationConflict:
  case DiagnosticCode::SchemaMismatch:
    throw std::invalid_argument(message);
  case DiagnosticCode::None:
  case DiagnosticCode::MissingRuntime:
  case DiagnosticCode::MissingRuntimeAuthority:
  case DiagnosticCode::MissingRuntimeContext:
  case DiagnosticCode::MissingHostVisibleConfigWeights:
  case DiagnosticCode::MissingRuntimeBuffer:
  case DiagnosticCode::GpuUnavailable:
  case DiagnosticCode::BatchSubmissionUnavailable:
  case DiagnosticCode::KernelCompilationFailed:
  case DiagnosticCode::OperationFailed:
    throw std::runtime_error(message);
  }
  throw std::runtime_error(message);
}

template <typename T>
[[nodiscard]] inline T unwrap_or_throw(Result<T> result,
                                       std::string_view surface) {
  if (result) {
    return std::move(*result);
  }
  throw_from_diagnostic(result.error());
  throw std::runtime_error(std::string(surface) +
                           ": unreachable after throw_from_diagnostic");
}

} // namespace tmnn
