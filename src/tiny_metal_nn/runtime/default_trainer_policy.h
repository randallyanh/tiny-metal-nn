#pragma once

#include "tiny-metal-nn/runtime_policy.h"

namespace tmnn::detail {

// The default trainer should honor the resolved context policy directly.
// Strict or fallback modes remain available, but they must be explicitly
// requested rather than silently imposed on the headline path.
inline BadStepRecoveryMode
resolve_default_trainer_recovery_mode(const RuntimePolicy &policy) {
  return policy.bad_step_recovery;
}

} // namespace tmnn::detail
