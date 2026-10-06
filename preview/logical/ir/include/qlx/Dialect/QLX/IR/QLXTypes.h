/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_QLX_QLXTYPES_H
#define QLX_DIALECT_QLX_QLXTYPES_H

#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/QLX/IR/QLXTypes.h.inc"

#endif // QLX_DIALECT_QLX_QLXTYPES_H
