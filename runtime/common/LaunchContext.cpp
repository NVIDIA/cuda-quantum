/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LaunchContext.h"
#include <stdexcept>

namespace cudaq::detail {

thread_local LaunchContext *pendingLaunchContext = nullptr;

LaunchContextHandoff::LaunchContextHandoff(LaunchContext &ctx) {
  if (pendingLaunchContext)
    throw std::logic_error(
        "Nested kernel launch: a launch context is already pending on this "
        "thread. Algorithms cannot be called before the kernel of an "
        "enclosing algorithm has been launched.");
  pendingLaunchContext = &ctx;
}

} // namespace cudaq::detail

cudaq::LaunchContext *cudaq::__cudaq_claimLaunchContext() {
  return detail::claimLaunchContext();
}
