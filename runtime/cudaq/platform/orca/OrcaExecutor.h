/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "OrcaServerHelper.h"
#include "common/ExecutionContext.h"
#include "common/RestClient.h"
// #include "cudaq.h"
#include "orca_qpu.h"

namespace cudaq::orca {

/// @brief The Executor provides an abstraction for executing compiled
/// quantum codes targeting a remote REST server. This type provides a
/// clean abstraction launching a vector of Jobs for sampling and observation
/// tasks, both synchronously and asynchronously.
class OrcaExecutor : public registry::RegisteredType<OrcaExecutor> {
protected:
  /// @brief The REST Client used to interact with the remote system
  RestClient client;

  /// @brief The ServerHelper, providing system-specific JSON-formatted
  /// job posts and results translation
  OrcaServerHelper *serverHelper;

  /// @brief The number of shots to execute
  std::size_t shots = 100;

public:
  OrcaExecutor() = default;
  virtual ~OrcaExecutor() = default;

  /// @brief Set the server helper
  void setServerHelper(OrcaServerHelper *helper) { serverHelper = helper; }

  /// @brief Set the number of shots to execute
  void setShots(std::size_t s) { shots = s; }

  /// @brief Execute the provided quantum codes and return a future object
  /// The caller can make this synchronous by just immediately calling .get().
  cudaq::details::future execute(TBIParameters params);
};

} // namespace cudaq::orca
