"""tiny_metal_nn — Apple Metal native runtime for hash-grid + MLP neural-field.

v1.0 design contract: docs/know-how/006-python-binding-design.md (v2).

Stage 4.1 status: skeleton — only `__version__` is exposed. Real API
(Trainer.from_config, training_step, inference, close, context manager)
lands in stages 4.2-4.5.
"""

from __future__ import annotations

# The pybind11 module (`_C`) sits next to this file once the wheel is
# installed (or once scikit-build-core has placed it in the editable
# install). Importing `_C` here makes its symbols available as
# `tiny_metal_nn.<symbol>` for users.
from ._C import (
    ClosedError,
    ConcurrentTrainingStepError,
    ConfigError,
    DTypeError,
    Trainer,
    __version__,
)

__all__ = [
    "ClosedError",
    "ConcurrentTrainingStepError",
    "ConfigError",
    "DTypeError",
    "Trainer",
    "__version__",
]
