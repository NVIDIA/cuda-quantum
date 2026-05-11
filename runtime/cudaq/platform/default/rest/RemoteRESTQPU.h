/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The RemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction.
class RemoteRESTQPU : public BaseRemoteRESTQPU {
public:
  RemoteRESTQPU() : BaseRemoteRESTQPU() {}
  RemoteRESTQPU(RemoteRESTQPU &&) = delete;
  ~RemoteRESTQPU() override;
};

} // namespace cudaq
