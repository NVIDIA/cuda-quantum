/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Executor.h"
#include "orca_qpu.h"

namespace cudaq {

/// @brief The Executor subclass for ORCA target which has a distinct sampling
/// API.
class OrcaExecutor : public Executor {
public:
  details::future execute(std::vector<KernelExecution> &codesToExecute,
                          cudaq::details::ExecutionContextType execType,
                          std::vector<char> *rawOutput) override {
    throw std::runtime_error(
        "ORCA backend does not support executing arbitrary kernels");
  }
  /// @brief Execute the provided ORCA quantum parameters and return a future
  /// object. The caller can make this synchronous by just immediately calling
  /// .get().
  details::future execute(cudaq::orca::TBIParameters params,
                          const std::string &kernelName);
};
} // namespace cudaq
