/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_DIALECT_QLX_TRANSFORMS_QLXVERIFYPBC_H
#define QLX_DIALECT_QLX_TRANSFORMS_QLXVERIFYPBC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace qlx {

/// Certify that a module is in Pauli-based-computation (PBC) normal form, the
/// contract `to_pbc` produces for downstream PBC lowering:
///   1. the body contains only prepare / pi/4 pauli_rotation / `mpp` /
///      discard / return / constant ops -- every Clifford has been absorbed;
///   2. every pauli_rotation is exactly +/- pi/4 (angle_pi_numer = +/-1,
///      angle_pi_denom = 4) -- the magic rotations;
///   3. no pauli_rotation follows a measurement (rotations precede measures);
///   4. the measured Pauli products pairwise commute (simultaneously
///      measurable).
/// On failure, a human-readable reason is written to `error`.
mlir::LogicalResult verifyPBCForm(mlir::ModuleOp module, std::string &error);

} // namespace qlx

#endif // QLX_DIALECT_QLX_TRANSFORMS_QLXVERIFYPBC_H
