/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef FABRIC_FABRICTYPES_H
#define FABRIC_FABRICTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricTypes.h.inc"

#endif // FABRIC_FABRICTYPES_H
