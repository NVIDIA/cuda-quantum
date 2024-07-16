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
/// server has specific capabilities.
struct RemoteCapabilities {
  /// True if the remote can perform state overlap operations.
  bool stateOverlap = false;
  /// True if the remote can perform serialized code execution (raw Python
  /// commands).
  bool serializedCodeExec = false;
  /// True if the remote can perform an entire VQE operation without and
  /// back-and-forth client/server communications.
  bool vqe = false;
  /// Constructor that broadcasts \p initValue to all fields.
  RemoteCapabilities(bool initValue)
      : stateOverlap(initValue), serializedCodeExec(initValue), vqe(initValue) {
  }
};

} // namespace cudaq
