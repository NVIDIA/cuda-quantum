/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraRemoteRESTQPU.h"

cudaq::QuEraRemoteRESTQPU::~QuEraRemoteRESTQPU() = default;

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::QuEraRemoteRESTQPU, quera)
