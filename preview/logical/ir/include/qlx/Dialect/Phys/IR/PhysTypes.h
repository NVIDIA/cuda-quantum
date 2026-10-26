/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_PHYSTYPES_H
#define QLX_DIALECT_PHYS_PHYSTYPES_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#include "qlx/Dialect/Event/IR/EventInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/Phys/IR/PhysTypes.h.inc"
#endif
