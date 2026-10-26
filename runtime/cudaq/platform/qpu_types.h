/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Umbrella header that makes all concrete QPU types visible.
/// Used by the typed platform validation (NVQPP_TARGET_QPU_TYPE).

#pragma once

#include "cudaq/platform/default/DefaultQPU.h"
#include "cudaq/platform/default/rest/RemoteRESTQPU.h"
#include "cudaq/platform/fermioniq/FermioniqQPU.h"
#include "cudaq/platform/mqpu/custatevec/GPUEmulatedQPU.h"
#include "cudaq/platform/orca/OrcaRemoteRESTQPU.h"
#include "cudaq/platform/pasqal/PasqalRemoteRESTQPU.h"
#include "cudaq/platform/quera/QuEraRemoteRESTQPU.h"
