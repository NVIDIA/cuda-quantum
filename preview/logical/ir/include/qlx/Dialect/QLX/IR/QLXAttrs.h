//===- QLXAttrs.h - QLX attribute declarations -----------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_QLX_QLXATTRS_H
#define QLX_DIALECT_QLX_QLXATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "qlx/Dialect/QLX/IR/QLXEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "qlx/Dialect/QLX/IR/QLXAttrs.h.inc"

#endif // QLX_DIALECT_QLX_QLXATTRS_H
