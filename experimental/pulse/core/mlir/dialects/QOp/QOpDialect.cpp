// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

using namespace mlir;

#include "cudaq-pulse/Dialect/QOp/QOpDialect.cpp.inc"

#include "cudaq-pulse/Dialect/QOp/QOpEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.cpp.inc"

namespace qop {

void QOpDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq-pulse/Dialect/QOp/QOpTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cudaq-pulse/Dialect/QOp/QOpOps.cpp.inc"
      >();
}

} // namespace qop
