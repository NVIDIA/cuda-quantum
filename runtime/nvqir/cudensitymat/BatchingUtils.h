
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/operators.h"

namespace cudaq {

namespace __internal__ {
// Helper to determine if the list of operators can be batched together
// for evolution.
bool checkBatchingCompatibility(
    const std::vector<sum_op<cudaq::matrix_handler>> &ops);
bool checkBatchingCompatibility(
    const std::vector<sum_op<cudaq::matrix_handler>> &hamOps,
    const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
        &listCollapseOps);
bool checkBatchingCompatibility(const std::vector<super_op> &listSuperOp);
bool checkBatchingCompatibility(
    const std::vector<cudaq::matrix_handler> &elemOps);
} // namespace __internal__
} // namespace cudaq
