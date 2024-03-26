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
class PyRestRemoteClient : public cudaq::BaseRemoteRestRuntimeClient {
public:
  /// @brief The constructor
  PyRestRemoteClient() : BaseRemoteRestRuntimeClient() {}
};

/// Implementation of QPU subtype that submits simulation request to NVCF.
/// REST client submitting jobs to NVCF-hosted `cudaq-qpud` service.
class PyNvcfRuntimeClient : public cudaq::BaseNvcfRuntimeClient {
public:
  /// @brief The constructor
  PyNvcfRuntimeClient() : BaseNvcfRuntimeClient() {}
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, PyRestRemoteClient, rest)
CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, PyNvcfRuntimeClient, NVCF)
