// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

using namespace mlir;

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.cpp.inc"

#include "cudaq-pulse/Dialect/Pulse/PulseEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.cpp.inc"

namespace pulse {

void PulseDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cudaq-pulse/Dialect/Pulse/PulseOps.cpp.inc"
      >();
}

} // namespace pulse
