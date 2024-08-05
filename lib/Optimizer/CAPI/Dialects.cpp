/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quake, quake, quake::QuakeDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CC, cc, cudaq::cc::CCDialect)
