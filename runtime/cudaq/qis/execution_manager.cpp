/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "cudaq/platform.h"

namespace cudaq {
bool __nvqpp__MeasureResultBoolConversion(int result, std::size_t id) {
  auto &platform = get_platform();
  auto *ctx = platform.get_exec_ctx();
  if (ctx && ctx->name == "tracer") {
    auto strId = std::to_string(id);
    // Only add to register names if we haven't already
    if (std::find(ctx->registerNames.begin(), ctx->registerNames.end(),
                  strId) == ctx->registerNames.end())
      ctx->registerNames.push_back(strId);
  }

  return result == 1;
}
} // namespace cudaq