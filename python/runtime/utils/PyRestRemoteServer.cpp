/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRestRemoteServer.h"

using namespace mlir;

namespace {
class PyBaseRestRemoteServer : public cudaq::BaseRemoteRestRuntimeServer {
public:
  /// @brief The constructor
  PyBaseRestRemoteServer() : BaseRemoteRestRuntimeServer() {}
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeServer, PyBaseRestRemoteServer, rest)
