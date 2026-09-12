/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_LVM_LVMOPS_H
#define QLX_DIALECT_LVM_LVMOPS_H
#include "qlx/Dialect/Event/IR/EventTypes.h"
#include "qlx/Dialect/LVM/IR/LVMAttrs.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/LVM/IR/LVMTypes.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#define GET_OP_CLASSES
#include "qlx/Dialect/LVM/IR/LVMOps.h.inc"

namespace qlx::lvm {

/// Verify that the inline placed body of `call` is the exact placement
/// refinement of its retained P0 callee. This is public so later dialects can
/// authenticate occurrence-level P1 provenance against the same invariant.
::mlir::LogicalResult verifyP0CallBodyRefinement(CallOp call);

} // namespace qlx::lvm
#endif
