/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// @brief A structure of boolean fields to indicate whether a given remote
/// server as specific capabilities.
/// @param stateOverlap True if the remote can perform state overlap operations.
/// @param serializedCodeExec True if the remote can perform serialized code
/// execution (raw Python commands)
/// @param vqe True if the remote can perform an entire VQE operation without
/// and back-and-forth client/server communications.
struct RemoteCapabilities {
  bool stateOverlap = false;
  bool serializedCodeExec = false;
  bool vqe = false;
  RemoteCapabilities(bool initValue)
      : stateOverlap(initValue), serializedCodeExec(initValue), vqe(initValue) {
  }
};

} // namespace cudaq
