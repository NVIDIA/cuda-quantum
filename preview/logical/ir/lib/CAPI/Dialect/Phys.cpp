/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Dialect/Phys.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "mlir/CAPI/Registration.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Phys, phys, qlx::phys::PhysDialect)
