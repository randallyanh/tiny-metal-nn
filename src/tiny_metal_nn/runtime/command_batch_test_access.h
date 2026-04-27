#pragma once

#include <string>

namespace tmnn::detail {

using CommandBatchErrorHook = std::string (*)(void *cmd_buf);

void set_command_batch_error_hook_for_testing(CommandBatchErrorHook hook);

} // namespace tmnn::detail
