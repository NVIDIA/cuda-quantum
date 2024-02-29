/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/ExecutionContext.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"

namespace cudaq {

/// @brief The Executor provides an abstraction for executing compiled
/// quantum codes targeting a remote REST server. This type provides a
/// clean abstraction launching a vector of Jobs for sampling and observation
/// tasks, both synchronously and asynchronously.
class Executor : public registry::RegisteredType<Executor> {
protected:
  /// @brief The REST Client used to interact with the remote system
  RestClient client;

  /// @brief The ServerHelper, providing system-specific JSON-formatted
  /// job posts and results translation
  ServerHelper *serverHelper;

  /// @brief The number of shots to execute
  std::size_t shots = 100;

public:
  Executor() = default;
  virtual ~Executor() = default;

  /// @brief Set the server helper
  void setServerHelper(ServerHelper *helper) { serverHelper = helper; }

  /// @brief Set the number of shots to execute
  void setShots(std::size_t s) { shots = s; }

  /// @brief Execute the provided quantum codes and return a future object
  /// The caller can make this synchronous by just immediately calling .get().
  details::future execute(std::vector<KernelExecution> &codesToExecute);
};

} // namespace cudaq
