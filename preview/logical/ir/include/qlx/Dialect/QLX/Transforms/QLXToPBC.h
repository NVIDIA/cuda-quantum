/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#ifndef QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H
#define QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace qlx {

/// Device-free P0 pass: rewrite a Clifford+T program into Pauli-based-
/// computation (PBC) form. Supported built-in Clifford actions are absorbed
/// into one symplectic stabilizer frame, so the program becomes an
/// exact weighted sequence of pi/4 Pauli-product rotations (emitted as
/// `qlx.apply #qlx.action<pauli_rotation>`) followed by terminal measurements
/// conjugated into Pauli products.
/// Repeat normalization and frame analysis precede full dialect conversion;
/// program and repeat patterns share the completed analysis by reference.
///
/// Static `cflow.repeat` regions remain folded when they carry distinct logical
/// qubits positionally with exact current SSA ownership, contain only
/// synthesized unitary actions, and close rotation support over the explicit
/// carry set. Counts through 64 may become one count-one phase chunk. Larger
/// nonidentity residuals require a full signed-Clifford period through 64 and
/// become a folded quotient plus optional count-one remainder. One
/// normalization may clone at most 4096 body operations. Workload-bearing idle
/// actions, logical-qubit program returns, unsupported periods or expansion,
/// other control flow, non-Clifford gates other than T/T-dagger (e.g. ccz),
/// local loop ownership, and mid-circuit measurement/feedforward are errors.
/// Rewrites the supplied module in place. On failure, the module may be
/// partially transformed and must not be used for further compilation.
mlir::LogicalResult lowerToPBC(mlir::ModuleOp module);

} // namespace qlx

#endif // QLX_DIALECT_QLX_TRANSFORMS_QLXTOPBC_H
