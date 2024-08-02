/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"

bool cudaq::__nvqpp__MeasureResultBoolConversion(int result) {
  auto &platform = get_platform();
  auto *ctx = platform.get_exec_ctx();
  if (ctx && ctx->name == "tracer")
    ctx->registerNames.push_back("");
  return result == 1;
}

namespace cudaq {
static ExecutionManager *execution_manager;

void setExecutionManagerInternal(ExecutionManager *em) {
  cudaq::info("external caller setting the execution manager.");
  execution_manager = em;
}

void resetExecutionManagerInternal() {
  cudaq::info("external caller clearing the execution manager.");
  execution_manager = nullptr;
}

ExecutionManager *getExecutionManagerInternal() { return execution_manager; }
} // namespace cudaq
