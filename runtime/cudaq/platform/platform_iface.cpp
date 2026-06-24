/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/platform_iface.h"
#include "cudaq/platform.h"

void cudaq::platform::with_execution_context(ExecutionContext &ctx,
                                             std::function<void()> f) {
  get_platform().with_execution_context(ctx, std::move(f));
}
