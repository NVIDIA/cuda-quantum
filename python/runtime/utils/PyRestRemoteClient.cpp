/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRestRemoteClient.h"

using namespace mlir;

namespace {
class PyRestRemoteClient : public cudaq::BaseRemoteRestRuntimeClient {
public:
  /// @brief The constructor
  PyRestRemoteClient() : BaseRemoteRestRuntimeClient() {}
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, PyRestRemoteClient, rest)
