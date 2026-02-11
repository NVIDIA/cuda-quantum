/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRemoteRESTQPU.h"
#include "common/RuntimeMLIR.h"

using namespace mlir;

namespace {

/// @brief The `RemoteRESTQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service. Specific
/// details about JSON payloads are abstracted via an abstract type called
/// ServerHelper, which is meant to be subtyped by each provided remote QPU
/// service. Moreover, this QPU handles launching kernels under a number of
/// Execution Contexts, including sampling and observation via synchronous or
/// asynchronous client invocations. This type should enable both QIR-based
/// backends as well as those that take OpenQASM2 as input.
class RemoteRESTQPU : public cudaq::BaseRemoteRESTQPU {

public:
  /// @brief The constructor
  RemoteRESTQPU() : BaseRemoteRESTQPU() {}

  RemoteRESTQPU(RemoteRESTQPU &&) = delete;
  virtual ~RemoteRESTQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteRESTQPU, remote_rest)
