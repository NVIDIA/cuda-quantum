/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Euler decompositions of one-qubit unitaries, shared by `squash-1q-to-u3`,
// which chooses a form and records it on a `quake.u3`, and the `U3ToRotations`
// decomposition pattern, which emits the recorded form.

#include "common/EigenDense.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <array>
#include <optional>

namespace cudaq::opt {

/// The discardable attribute on a `quake.u3` that hints the preferred Euler
/// decomposition, as chosen by `squash-1q-to-u3` (`"zxz"` or `"xzx"`).
inline constexpr llvm::StringLiteral eulerFormAttrName = "euler";

/// The represented matrix is
/// `exp(i*phase) R_outer(angles[2]) R_middle(angles[1]) R_outer(angles[0])`.
struct EulerDecomposition {
  cudaq::quake::Pauli outer;
  cudaq::quake::Pauli middle;
  std::array<double, 3> angles;
  double phase;

  /// Decompose `u` into rotations about `outerAxis`, `middleAxis`, and
  /// `outerAxis`, which must be ZXZ or XZX. Angles within roundoff of a
  /// multiple of pi/4, zero included, are replaced by that exact multiple.
  static EulerDecomposition get(const Eigen::Matrix2cd &u,
                                cudaq::quake::Pauli outerAxis,
                                cudaq::quake::Pauli middleAxis);

  /// The number of nonzero rotations `emit` creates.
  unsigned numRotations() const;

  /// The `u3(theta, phi, lambda)` angles of the represented matrix, up to a
  /// global phase. Only valid for the ZXZ form. Angles within roundoff of a
  /// multiple of pi/4 are replaced by that exact multiple.
  std::array<double, 3> toU3Angles() const;

  /// The name of this form for `eulerFormAttrName`.
  llvm::StringRef formName() const;

  /// Emit the nonzero rotations and an exact phase correction on the scalar
  /// `input` wire or reference, and return the wire it produces (or `input`
  /// for a reference).
  mlir::Value emit(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value input) const;
};

/// Parse an Euler form name. Returns the (outer, middle) axes, or nothing for
/// an unknown name.
std::optional<std::pair<cudaq::quake::Pauli, cudaq::quake::Pauli>>
parseEulerForm(llvm::StringRef name);

/// Snap `angle`, which a pass computed with an error of a few ulps, to an exact
/// multiple of pi/4 when it is within roundoff of one.
double snapAngleToQuarterTurn(double angle);

} // namespace cudaq::opt
