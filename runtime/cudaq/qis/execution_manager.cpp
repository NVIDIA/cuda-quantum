/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "common/PluginUtils.h"

namespace cudaq {
static ExecutionManager *execution_manager;

void setExecutionManagerInternal(ExecutionManager *em) {
  CUDAQ_INFO("external caller setting the execution manager.");
  execution_manager = em;
}

void resetExecutionManagerInternal() {
  CUDAQ_INFO("external caller clearing the execution manager.");
  execution_manager = nullptr;
}

ExecutionManager *getExecutionManagerInternal() { return execution_manager; }

} // namespace cudaq
