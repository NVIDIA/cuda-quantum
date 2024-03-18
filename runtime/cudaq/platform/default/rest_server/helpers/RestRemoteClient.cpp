/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRestRemoteClient.h"

using namespace mlir;

namespace {
class RemoteRestRuntimeClient : public cudaq::BaseRemoteRestRuntimeClient {
public:
  /// @brief The constructor
  RemoteRestRuntimeClient() : BaseRemoteRestRuntimeClient() {}
};

/// REST client submitting jobs to NVCF-hosted `cudaq-qpud` service.
class NvcfRuntimeClient : public cudaq::BaseNvcfRuntimeClient {
public:
  /// @brief The constructor
  NvcfRuntimeClient() : BaseNvcfRuntimeClient() {}
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, RemoteRestRuntimeClient, rest)
CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, NvcfRuntimeClient, NVCF)
