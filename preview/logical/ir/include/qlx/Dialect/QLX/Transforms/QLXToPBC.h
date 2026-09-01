/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H
#define QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace qlx {

/// Device-free P0 pass: rewrite a Clifford+T program into Pauli-based-
/// computation (PBC) form. Every Clifford (H/S/S-dagger/X/Y/Z/SX/CX/CZ) is
/// commuted to the end via one symplectic stabilizer frame, so the program
/// becomes a sequence of pi/4 Pauli-product rotations (one per T/T-dagger,
/// emitted as `qlx.apply #qlx.action<pauli_rotation>` with angle pi/4) followed
/// by the terminal measurements conjugated into Pauli products.
///
/// Scope of this pass: single-block programs with terminal measurements. State
/// preparation is limited to |0>, |1>, |+>, |->; measurements to the Z and X
/// bases. Non-Clifford gates other than T/T-dagger (e.g. CCZ), mid-circuit
/// measurement with feed-forward, and control flow are reported as errors.
mlir::LogicalResult lowerToPBC(mlir::ModuleOp module);

} // namespace qlx

#endif // QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H
