# tmnn error handling

`tiny-metal-nn` now uses a single public non-throwing contract:

```cpp
tmnn::Result<T> = std::expected<T, tmnn::DiagnosticInfo>
```

Use the `try_*` factory/runtime helpers when you want structured failures
instead of exceptions:

```cpp
auto result = tmnn::try_create_from_config(
    3, 1, config, tmnn::default_trainer_config(), ctx);
if (!result) {
  std::cerr << tmnn::format_diagnostic(result.error()) << "\n";
  for (const auto& detail : result.error().details) {
    std::cerr << "  " << tmnn::format_diagnostic_detail(detail) << "\n";
  }
  return;
}

tmnn::Trainer trainer = std::move(*result);
```

The throwing wrappers (`create_from_config(...)`, `create_trainer(...)`, and
similar) remain available. They are now thin wrappers over the same result path
and throw from the returned `DiagnosticInfo`.

## Diagnostic payload

`DiagnosticInfo` contains:

- `code` — stable machine-readable `DiagnosticCode`
- `operation` — high-level API surface that failed
- `message` — short human-readable summary
- `details` — optional structured detail list, commonly used for JSON/config
  validation findings

`DiagnosticDetail` / `ConfigDiagnostic` contains:

- `severity` — `Info`, `Warning`, or `Error`
- `path` — config/runtime field path when available
- `message` — the detail text

## Logger hook

Install a process-wide hook to observe diagnostics centrally:

```cpp
tmnn::set_logger_hook([](const tmnn::DiagnosticInfo& diagnostic) {
  std::cerr << tmnn::format_diagnostic(diagnostic) << "\n";
});
```

Notes:

- the hook is optional; unset it with `tmnn::clear_logger_hook()`
- failure results emitted through `make_error_result(...)` reach the hook
- config canonicalization notes can also reach the hook with
  `DiagnosticCode::None` and populated `details`

## Diagnostic codes

| Code | Meaning |
| --- | --- |
| `None` | Informational logger-only event; not a failure |
| `InvalidArgument` | Input/config was malformed or failed validation |
| `NullObject` | Required public object pointer was null |
| `MissingRuntime` | `Trainer` has no bound runtime |
| `MissingRuntimeAuthority` | Runtime-backed evaluator/runtime authority is absent |
| `MissingRuntimeContext` | A required `MetalContext` is absent |
| `MissingHostVisibleConfigWeights` | Evaluator creation requires host-visible config weights that are unavailable |
| `MissingRuntimeBuffer` | A required Metal/runtime buffer is missing |
| `GpuUnavailable` | Metal GPU execution is unavailable on the current context/device |
| `BatchSubmissionUnavailable` | Command-batch submission resources could not be acquired |
| `KernelCompilationFailed` | Required MSL pipeline/kernel compilation failed |
| `UnsupportedOperation` | Requested surface exists, but the current runtime does not support it |
| `UnsupportedModelType` | The default runtime cannot lower the supplied descriptor/model shape |
| `UnsupportedLossType` | The default runtime cannot lower the supplied loss object/config |
| `ConfigurationConflict` | Two public inputs disagree semantically (for example explicit loss vs `TrainerConfig`) |
| `SchemaMismatch` | Adapter/model or schema/runtime dimensions do not match |
| `OperationFailed` | Fallback catch-all for other runtime/construction failures |

## Sample exit codes

The flagship sample `samples/mlp_learning_an_image.cpp` prints explicit
`exit_code=...` values:

| Exit code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | No Metal-capable GPU available |
| `3` | Config file open failed |
| `4` | Config parse failed |
| `5` | Failed to write `reference.ppm` |
| `6` | Non-finite training loss detected |
| `7` | Inference failure |
| `8` | Failed to write `learned.ppm` |
| `9` | Trainer construction from config failed |
| `64` | Uncaught exception escaped `main()` |
