/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_H
#define CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>

namespace cudaq::logical {

/// A semantic commitment made by a canonical CUDA-Q Logical root. Stage is
/// derived from the root operation; it is not independent state stored
/// throughout the IR.
enum class Stage : std::uint8_t { P0, P1, P2, P3, P4 };

/// The role of a canonical executable root. These values describe semantic
/// ownership and deliberately do not mirror dialect namespaces.
enum class RootKind : std::uint8_t {
  LogicalProgram,
  PlacedKernel,
  QECCircuit,
  QECGadget,
  QECProtocol,
  PhysicalGraph,
  RealtimePlan
};

llvm::StringRef stringifyStage(Stage stage);
llvm::StringRef stringifyRootKind(RootKind kind);

} // namespace cudaq::logical

#include "CUDAQLogical/Interfaces/SemanticInterfaces.h.inc"

#endif // CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_H
