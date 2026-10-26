/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#ifndef QLX_DIALECT_QLX_TRANSFORMS_QLXSYNTHESIZE_H
#define QLX_DIALECT_QLX_TRANSFORMS_QLXSYNTHESIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace qlx {

/// Device-free P0 legalization to positive H/S/T/CX. Finite built-in actions
/// are decomposed exactly; static Pauli-product rotations use basis changes, a
/// parity ladder, and the native Ross-Selinger synthesizer. Each rotation uses
/// its authored `precision` when present, falling back to the call-wide
/// precision. Dynamic rotations fail closed.
mlir::LogicalResult synthesizeRotations(mlir::ModuleOp module,
                                        double precision);

/// Verify that every remaining `qlx.apply` is in the positive H/S/T/CX basis
/// (or is the semantic no-op `idle`). Emits an operation diagnostic on failure.
mlir::LogicalResult verifyCliffordT(mlir::ModuleOp module);

} // namespace qlx

#endif // QLX_DIALECT_QLX_TRANSFORMS_QLXSYNTHESIZE_H
