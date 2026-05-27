/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class Operation;
class Block;
class Region;
} // namespace mlir

namespace cudaq::quake::detail {
struct UnitaryOpGroup {
  UnitaryOpGroup() = default;

  mlir::Block *block = nullptr;
  mlir::Operation *firstOp = nullptr;
  mlir::Operation *lastOp = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
};

using UnitaryOpGroups = llvm::SmallVector<cudaq::quake::detail::UnitaryOpGroup>;

/// Analysis to group unitary operations within a block together.
struct UnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryOpGroupingAnalysis)

  explicit UnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  const UnitaryOpGroups &getGroups() const { return groups; }

private:
  void performAnalysis(mlir::Operation *operation);
  void scanRegion(mlir::Region &region);
  void scanBlock(mlir::Block &block);

  UnitaryOpGroups groups;
};
} // namespace cudaq::quake::detail
