/**
 * @file src/python/tmnn_pybind.cpp
 * @brief tiny_metal_nn Python binding (pybind11). Stage 4 deliverable.
 *
 * Design contract:
 *   - 006 v2 (API shape, error message format, anti-patterns)
 *   - 007    (GIL release boundary, lifetime invariants, torture suite)
 *   - 011    (JSON schema accepted by Trainer.from_config)
 *
 * The Python-facing module name is `tiny_metal_nn._C`. The user-visible
 * surface lives at `tiny_metal_nn` (see `tiny_metal_nn/__init__.py`).
 *
 * Stage progress (006 v2 §11):
 *   4.1 ✓ skeleton + __version__
 *   4.2 ✓ Trainer.from_config
 *   4.3 ✓ training_step numpy I/O + GIL release
 *   4.4 ✓ inference owned-output buffer
 *   4.5 ✓ close + __enter__ / __exit__ + ClosedError
 */

#include "tiny-metal-nn/factory_json.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/trainer.h"

#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

namespace {

// 006 v2 §7.5 — three-segment error format helper.
// Produces:
//     <head>
//       In:     <where>
//       Common: <common>
//       See:    <see>
// `common` and `see` may be empty; sections are skipped if so.
std::string format_three_segment(const std::string &head,
                                 const std::string &where,
                                 const std::string &common = "",
                                 const std::string &see = "") {
  std::ostringstream s;
  s << head;
  if (!where.empty()) {
    s << "\n  In:     " << where;
  }
  if (!common.empty()) {
    s << "\n  Common: " << common;
  }
  if (!see.empty()) {
    s << "\n  See:    " << see;
  }
  return s.str();
}

// Common-values lookup for known config paths. Used by ConfigError
// formatting so users see "what should this field be" alongside the
// diagnostic. Paths missing from the table get a generic See: link.
const std::unordered_map<std::string, std::string> kConfigPathCommon = {
    {"encoding.otype", "HashGrid, RotatedMHE"},
    {"encoding.n_levels", "16 (instant-NGP default)"},
    {"encoding.n_features_per_level", "2 (instant-NGP), 4 (high-capacity)"},
    {"encoding.log2_hashmap_size", "19 (instant-NGP), 14 (small / unit test)"},
    {"encoding.base_resolution", "16.0 (instant-NGP default)"},
    {"encoding.per_level_scale", "1.447 (instant-NGP default), 1.5 (small)"},
    {"encoding.interpolation", "Linear (only one supported in v1.0)"},
    {"network.otype", "FullyFusedMLP (only one supported in v1.0)"},
    {"network.n_neurons", "16, 32, 64, 128"},
    {"network.n_hidden_layers", "1, 2, 3, 4"},
    {"network.activation",
     "ReLU (only one supported in v1.0; non-ReLU activations on the v1.x roadmap)"},
    {"network.output_activation", "None (linear identity) or Linear (alias)"},
    {"loss.otype", "L2, L1, Huber, Cosine"},
    {"loss.huber_delta", "1.0 (default), 0.5, 0.1 (sharp)"},
    {"optimizer.otype", "Adam (only one supported in v1.0)"},
    {"optimizer.learning_rate", "1e-3 (default), 1e-2 (instant-NGP), 1e-4 (fine-tune)"},
    {"optimizer.beta1", "0.9 (default)"},
    {"optimizer.beta2", "0.99 (default), 0.999 (PyTorch default)"},
    {"optimizer.epsilon", "1e-15 (default)"},
    {"weight_init.hash_grid_init", "Uniform (default), Zero"},
    {"weight_init.mlp_init",
     "KaimingUniform (default), KaimingNormal, XavierUniform, "
     "XavierNormal, Uniform, Normal, Zero"},
    {"weight_init.mlp_nonlinearity",
     "ReLU (default), Linear, LeakyReLU, Tanh, Sigmoid"},
    {"weight_init.seed", "42 (default), any 64-bit integer"},
    {"batch_size", "1024 (default), 4096, 8192"},
};

// Public migration guide path used as the See: link for config / dtype
// errors. Lives in the user-facing docs/, not in gitignored know-how/.
constexpr const char *kSeeMigrationGuide =
    "docs/TCNN-MIGRATION-GUIDE.md § 10";

// Custom C++ exception types. Required because `register_exception` matches
// the registered type AND its subclasses; using a stdlib base like
// std::runtime_error would cause pybind11's own builtins (value_error,
// type_error) to be incorrectly translated to one of these.
class TmnnClosedError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class TmnnConcurrentTrainingStepError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// 006 v2 §7.4 typed exception hierarchy. ConfigError inherits Python
// ValueError so existing user code catching ValueError keeps working;
// DTypeError inherits TypeError for the same reason.
class TmnnConfigError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class TmnnDTypeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};


// Convert a Python dict (or any JSON-serializable Python value) to
// nlohmann::json by going through Python's `json.dumps`. Slower than a
// type-by-type walk, but trivially correct and the cost falls only on
// `from_config` which is a cold path.
nlohmann::json py_to_json(py::handle obj) {
  py::module_ json_mod = py::module_::import("json");
  const std::string s = json_mod.attr("dumps")(obj).cast<std::string>();
  return nlohmann::json::parse(s);
}

// Validate a numpy array for tmnn input (float32 + C-contiguous). Throws
// tmnn.DTypeError on dtype mismatch and ValueError on non-contiguous
// storage. Both messages follow the 006 v2 §7.5 three-segment format.
void check_float_carray(const py::array &arr, const char *name) {
  if (!arr.dtype().is(py::dtype::of<float>())) {
    const std::string head = std::string(name) + " must be float32, got " +
                             py::str(arr.dtype()).cast<std::string>();
    throw TmnnDTypeError(format_three_segment(
        head, std::string("training_step / inference ") + name + " argument",
        ".astype(np.float32) on a numpy array, "
        "or torch.tensor(..., dtype=torch.float32)",
        kSeeMigrationGuide));
  }
  if (!(arr.flags() & py::array::c_style)) {
    const std::string head =
        std::string(name) + " must be C-contiguous";
    throw py::value_error(format_three_segment(
        head, std::string("training_step / inference ") + name + " argument",
        ".contiguous() on a torch tensor, or np.ascontiguousarray(...)",
        kSeeMigrationGuide));
  }
}

// Coerce a Python object to a numpy array. Accepts numpy arrays directly
// and array-likes (e.g., torch.Tensor on CPU) via the __array__ protocol.
//
// Non-CPU torch tensors are rejected explicitly rather than silently
// staged through CPU (006 v2 §3.2: no surprising copies). The user must
// call .cpu() or wait for v1.x+ MPS borrowed-input.
py::array coerce_to_numpy(py::handle obj, const char *name) {
  // Torch tensors expose `.device`; non-CPU devices need an explicit copy
  // we don't want to do silently. Numpy arrays don't have `.device`.
  if (py::hasattr(obj, "device") && py::hasattr(obj, "cpu")) {
    const std::string dev =
        py::str(obj.attr("device")).cast<std::string>();
    if (dev.find("cpu") == std::string::npos) {
      // 006 v2 §7.5 three-segment format. The See: link points at the
      // public migration guide (gitignored docs are not user-visible).
      const std::string head = std::string(name) +
                               ": tensor is on device '" + dev +
                               "'; tmnn v1.0 accepts CPU tensors only";
      throw py::value_error(format_three_segment(
          head, std::string("training_step / inference ") + name + " argument",
          ".cpu() on the tensor before passing in (zero-copy MPS interop "
          "is on the v1.x roadmap)",
          kSeeMigrationGuide));
    }
  }

  // py::array::ensure runs the array protocol (__array__) if needed and
  // returns an empty py::array on failure (false in bool context).
  auto arr = py::array::ensure(obj);
  if (!arr) {
    throw py::type_error(
        std::string(name) +
        " must be a numpy array or array-like (e.g., torch CPU tensor).");
  }
  return arr;
}

// Wrapper that adds a `closed` flag on top of the move-only C++ Trainer.
// 006 v2 §3 / 007 §2.3: explicit close() + context manager are part of
// the public Python contract; raw tmnn::Trainer has no such state. The
// inner unique_ptr is reset on close() so GPU resources are released
// deterministically without waiting for Python GC.
struct PyTrainer {
  std::unique_ptr<tmnn::Trainer> inner;
  bool closed = false;
  // unique_ptr indirection because std::atomic<bool> is not move-constructible
  // and PyTrainer must stay movable (from_config returns by value).
  // Per 007 §1.4: protects against concurrent training_step calls from
  // multiple Python threads — the C++ trainer is single-threaded by design.
  std::unique_ptr<std::atomic<bool>> in_flight =
      std::make_unique<std::atomic<bool>>(false);

  PyTrainer() = default;
  PyTrainer(PyTrainer &&) noexcept = default;
  PyTrainer &operator=(PyTrainer &&) noexcept = default;
  PyTrainer(const PyTrainer &) = delete;
  PyTrainer &operator=(const PyTrainer &) = delete;

  // Throws TmnnClosedError when post-close calls are made; pybind11's
  // registered translator (PYBIND11_MODULE below) maps that to the
  // Python class `tiny_metal_nn._C.ClosedError`.
  void check_open() const {
    if (closed) {
      throw TmnnClosedError(
          "trainer is closed (call from_config to make a new one)");
    }
  }

  void close() {
    if (closed) {
      return;  // double-close is a no-op (006 v2 / 007 §2.3 contract)
    }
    if (inner) {
      // ~Trainer() already runs sync_weights+drain in its dtor (stage 2),
      // but reset() here makes the GPU release deterministic relative to
      // the user's `with`-block exit instead of waiting on Python GC.
      try {
        inner->sync_weights();
      } catch (...) {
      }
      inner.reset();
    }
    closed = true;
  }
};

} // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() =
      "tiny_metal_nn native binding.\n"
      "Public surface lives at tiny_metal_nn.*; this module is internal.\n"
      "See docs/know-how/006-python-binding-design.md v2 for the design contract.";
  m.attr("__version__") = "0.1.0a2";

  // ── Typed exception hierarchy (006 v2 §7.4) ──────────────────────────
  // Each bound to its dedicated C++ subclass so pybind11's own value_error /
  // type_error are not caught by these translators. ConfigError inherits
  // Python ValueError; DTypeError inherits TypeError; the others inherit
  // RuntimeError. This keeps `except ValueError` / `except TypeError`
  // user code working while letting power users catch the typed forms.
  py::register_exception<TmnnClosedError>(m, "ClosedError");
  py::register_exception<TmnnConcurrentTrainingStepError>(
      m, "ConcurrentTrainingStepError");
  py::register_exception<TmnnConfigError>(m, "ConfigError", PyExc_ValueError);
  py::register_exception<TmnnDTypeError>(m, "DTypeError", PyExc_TypeError);

  // ── tmnn.Trainer (Python-side wrapper around C++ tmnn::Trainer) ──────
  py::class_<PyTrainer>(m, "Trainer", R"pbdoc(
The single object for training and inference on hash-grid + MLP models.

Construct via :py:meth:`Trainer.from_config`. Use as a context manager
(``with tmnn.Trainer.from_config(...) as t: ...``) or call ``close()``
explicitly to deterministically release GPU resources.
)pbdoc")
      .def_static(
          "from_config",
          [](py::object config, int n_input, int n_output) -> PyTrainer {
            const nlohmann::json json_config = py_to_json(config);
            auto result = tmnn::try_create_from_config(
                static_cast<uint32_t>(n_input),
                static_cast<uint32_t>(n_output), json_config);
            if (!result.has_value()) {
              // 006 v2 §7.4 typed exception; inherits Python ValueError so
              // existing `except ValueError` user code still catches it.
              // §7.5 three-segment format: pull the first error detail's
              // path so we can attach a Common: hint from the lookup
              // table.
              const auto &diag = result.error();
              std::string head = diag.message;
              std::string where;
              std::string common;
              if (!diag.details.empty()) {
                const auto &first = diag.details.front();
                where = first.path;
                auto it = kConfigPathCommon.find(first.path);
                if (it != kConfigPathCommon.end()) {
                  common = it->second;
                }
              }
              throw TmnnConfigError(format_three_segment(
                  head, where, common, kSeeMigrationGuide));
            }
            PyTrainer self;
            self.inner =
                std::make_unique<tmnn::Trainer>(std::move(*result));
            return self;
          },
          py::arg("config"), py::kw_only(), py::arg("n_input"),
          py::arg("n_output"),
          R"pbdoc(
Construct a Trainer from a JSON-shaped config dict.

Args:
    config: dict matching the schema in
        ``docs/know-how/011-json-schema-frozen.md``. Top-level keys
        ``encoding``, ``network``, ``loss``, ``optimizer``,
        ``weight_init``, and ``batch_size`` are accepted.
    n_input: input dimensionality (positive integer).
    n_output: output dimensionality (positive integer).

Raises:
    ValueError: if any field fails schema validation. Message format
        follows the three-segment structure in 006 v2 §7.5.
)pbdoc")
      .def("step",
           [](const PyTrainer &self) {
             self.check_open();
             return self.inner->step();
           },
           "Current training step counter (0 if no training_step called).")
      .def("is_gpu_available",
           [](const PyTrainer &self) {
             self.check_open();
             return self.inner->is_gpu_available();
           },
           "True iff the underlying runtime has an available Metal device.")
      .def(
          "training_step",
          [](PyTrainer &self, py::object input_obj,
             py::object target_obj) -> float {
            self.check_open();
            py::array input = coerce_to_numpy(input_obj, "input");
            py::array target = coerce_to_numpy(target_obj, "target");
            check_float_carray(input, "input");
            check_float_carray(target, "target");

            if (input.ndim() != 2) {
              throw py::value_error(
                  "input must be 2D (batch_size, n_input_dims)");
            }
            if (target.ndim() != 2) {
              throw py::value_error(
                  "target must be 2D (batch_size, n_output_dims)");
            }
            if (input.shape(0) != target.shape(0)) {
              throw py::value_error(
                  "input and target must have matching batch_size");
            }

            // 007 §1.4: detect concurrent training_step calls from another
            // Python thread. Single-shot atomic flip; failure throws the
            // dedicated exception so the user sees the misuse explicitly
            // rather than getting silently serialized (a mutex would mask
            // the bug; an atomic flag teaches).
            bool expected = false;
            if (!self.in_flight->compare_exchange_strong(expected, true)) {
              throw TmnnConcurrentTrainingStepError(
                  "training_step is already in progress on another thread. "
                  "tmnn.Trainer is single-threaded; use one Trainer per "
                  "thread.");
            }
            struct InFlightGuard {
              std::atomic<bool> &flag;
              ~InFlightGuard() { flag.store(false); }
            } guard{*self.in_flight};

            const int N = static_cast<int>(input.shape(0));
            const float *input_ptr = static_cast<const float *>(input.data());
            const float *target_ptr =
                static_cast<const float *>(target.data());

            // 007 §1.2: manual GIL release boundary. Argument parsing
            // (above) ran with GIL held; we release here for the GPU
            // dispatch + sync. py::call_guard would have parsed under
            // release, which is unsafe for py::array touches.
            tmnn::TrainingStepResult result;
            {
              py::gil_scoped_release release;
              result = self.inner->training_step(input_ptr, target_ptr, N);
            }
            return result.loss;
          },
          py::arg("input"), py::arg("target"),
          R"pbdoc(
Run one fused training step (forward + backward + optimizer).

Blocks until the GPU command buffer completes; async dispatch is on
the v1.x+ roadmap (006 v2 §12).

Args:
    input: float32 C-contiguous array of shape (N, n_input_dims).
    target: float32 C-contiguous array of shape (N, n_output_dims).

Returns:
    float: the loss value for this step.

Raises:
    TypeError: if input or target is not float32.
    ValueError: if shapes mismatch or the array is non-contiguous.
    ClosedError: if the trainer was already closed.
)pbdoc")
      .def(
          "inference",
          [](PyTrainer &self, py::object input_obj) -> py::array_t<float> {
            self.check_open();
            py::array input = coerce_to_numpy(input_obj, "input");
            check_float_carray(input, "input");
            if (input.ndim() != 2) {
              throw py::value_error(
                  "input must be 2D (batch_size, n_input_dims)");
            }

            // Same in_flight guard as training_step: inference touches the
            // same param store / pool the training side mutates, so a
            // concurrent training_step + inference is undefined behavior
            // (007 §1.4 strict reading). Share the flag.
            bool expected = false;
            if (!self.in_flight->compare_exchange_strong(expected, true)) {
              throw TmnnConcurrentTrainingStepError(
                  "inference called while another thread is in "
                  "training_step / inference. tmnn.Trainer is "
                  "single-threaded; use one Trainer per thread.");
            }
            struct InFlightGuard {
              std::atomic<bool> &flag;
              ~InFlightGuard() { flag.store(false); }
            } guard{*self.in_flight};

            const int N = static_cast<int>(input.shape(0));
            const auto plan = self.inner->batch_plan();
            const auto output_dims =
                static_cast<py::ssize_t>(plan.target_dims);

            // 007 §2.1 scenario B: the returned array owns its own buffer.
            // py::array_t with a shape allocates fresh memory; when this
            // function returns, that buffer is bound to the numpy array,
            // not to `self`. The user can `del trainer` and the output
            // stays valid.
            py::array_t<float> output(
                {static_cast<py::ssize_t>(N), output_dims});
            auto out_buf = output.request();
            float *output_ptr = static_cast<float *>(out_buf.ptr);
            const float *input_ptr =
                static_cast<const float *>(input.data());

            bool ok;
            {
              py::gil_scoped_release release;
              ok = self.inner->inference(input_ptr, output_ptr, N);
            }
            if (!ok) {
              throw std::runtime_error(
                  "inference failed (see trainer diagnostics for details)");
            }
            return output;
          },
          py::arg("input"),
          R"pbdoc(
Evaluate the trained model at the given input positions.

Returns a freshly-allocated numpy array whose lifetime is independent
of the trainer (007 §2.1 scenario B).

Args:
    input: float32 C-contiguous array of shape (N, n_input_dims).

Returns:
    numpy.ndarray: float32 array of shape (N, n_output_dims).

Raises:
    TypeError: if input is not float32.
    ValueError: if input is not 2D or non-contiguous.
    ClosedError: if the trainer was already closed.
)pbdoc")
      .def(
          "__repr__",
          [](const PyTrainer &self) -> std::string {
            if (self.closed) {
              return "<tmnn.Trainer closed>";
            }
            const auto &model = self.inner->model();
            const auto plan = self.inner->batch_plan();
            // 006 v2 §3.4: repr should be informative. Dynamic-cast to
            // NetworkWithInputEncoding to split into encoding + network
            // when possible; fall back to the bare module summary if
            // the user has plugged in a non-canonical model.
            std::ostringstream s;
            s << "tmnn.Trainer(\n";
            const auto *composed =
                dynamic_cast<const tmnn::NetworkWithInputEncoding *>(&model);
            if (composed) {
              const auto &enc = composed->encoding();
              const auto &net = composed->network();
              s << "  encoding=" << enc->name() << "(in="
                << enc->n_input_dims() << ", out=" << enc->n_output_dims()
                << ", params=" << enc->n_params() << "),\n"
                << "  network=" << net->name() << "(in="
                << net->n_input_dims() << ", out=" << net->n_output_dims()
                << ", params=" << net->n_params() << "),\n";
            } else {
              s << "  model=" << model.name() << "(in="
                << model.n_input_dims() << ", out=" << model.n_output_dims()
                << ", params=" << model.n_params() << "),\n";
            }
            s << "  step=" << self.inner->step() << ",\n"
              << "  batch_size=" << plan.max_batch_size << ",\n"
              << "  device=metal,\n"
              << ")";
            return s.str();
          },
          R"pbdoc(
Multi-line summary of the trainer's model + state. See ``summary()``
for the verbose form.
)pbdoc")
      .def(
          "summary",
          [](const PyTrainer &self, int batch_size) -> std::string {
            if (self.closed) {
              return "<tmnn.Trainer closed>";
            }
            const auto &model = self.inner->model();
            const auto plan = self.inner->batch_plan();
            std::ostringstream s;
            s << "tmnn.Trainer\n";
            const auto *composed =
                dynamic_cast<const tmnn::NetworkWithInputEncoding *>(&model);
            if (composed) {
              const auto &enc = composed->encoding();
              const auto &net = composed->network();
              s << "  Encoding:        " << enc->name() << "\n"
                << "    Input dims:    " << enc->n_input_dims() << "\n"
                << "    Output dims:   " << enc->n_output_dims()
                << "  (-> network input)\n"
                << "    Parameters:    " << enc->n_params() << "\n"
                << "  Network:         " << net->name() << "\n"
                << "    Input dims:    " << net->n_input_dims() << "\n"
                << "    Output dims:   " << net->n_output_dims() << "\n"
                << "    Parameters:    " << net->n_params() << "\n";
            } else {
              s << "  Model:           " << model.name() << "\n"
                << "    Input dims:    " << model.n_input_dims() << "\n"
                << "    Output dims:   " << model.n_output_dims() << "\n"
                << "    Parameters:    " << model.n_params() << "\n";
            }
            s << "  Total params:    " << model.n_params() << " (fp32)\n"
              << "  Current step:    " << self.inner->step() << "\n"
              << "  Max batch size:  " << plan.max_batch_size << "\n"
              << "  Device:          metal\n";
            if (batch_size > 0 &&
                static_cast<uint32_t>(batch_size) > plan.max_batch_size) {
              s << "  Note: requested batch_size=" << batch_size
                << " exceeds max_batch_size=" << plan.max_batch_size
                << "; pass batch_size in from_config to widen the plan.\n";
            }
            return s.str();
          },
          py::arg("batch_size") = 0,
          R"pbdoc(
Detailed multi-line trainer summary. Pass ``batch_size`` to flag if a
specific batch size would exceed the trained plan.
)pbdoc")
      .def("close", &PyTrainer::close,
           R"pbdoc(
Synchronize pending GPU work and release the underlying Metal resources.

Idempotent: calling close() a second time is a no-op. After close, any
training_step / inference / step / is_gpu_available call raises
ClosedError. Prefer the ``with`` statement to closing manually.
)pbdoc")
      .def("closed",
           [](const PyTrainer &self) { return self.closed; },
           "True iff close() has been called (or the context manager exited).")
      .def("__enter__",
           [](py::object self) { return self; },
           "Context-manager entry. Returns self.")
      .def(
          "__exit__",
          [](PyTrainer &self, py::object, py::object, py::object) {
            self.close();
            return false;  // don't suppress exceptions raised inside `with`
          },
          "Context-manager exit. Calls close() and propagates any raised "
          "exception.");
}
