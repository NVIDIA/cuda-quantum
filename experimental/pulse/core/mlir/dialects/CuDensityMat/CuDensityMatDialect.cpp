// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

using namespace mlir;

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.cpp.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.cpp.inc"

namespace cudm {

void CuDensityMatDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.cpp.inc"
      >();
}

} // namespace cudm
