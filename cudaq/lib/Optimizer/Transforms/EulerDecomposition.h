/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Euler decompositions of one-qubit unitaries, shared by `fuse-1q-to-u3`,
// which uses them for decomposition cost estimatations, and the `U3ToRotations`
// decomposition pattern.

#include "common/EigenDense.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <array>
#include <optional>

namespace cudaq::opt {

/// The discard-able attribute on a `quake.u3` that hints the preferred Euler
/// decomposition, as chosen by `fuse-1q-to-u3` (`"zxz"` or `"xzx"`).
inline constexpr llvm::StringLiteral eulerFormAttrName = "euler";

/// The represented matrix is
/// `exp(i*phase) R_outer(angles[2]) R_middle(angles[1]) R_outer(angles[0])`.
struct EulerDecomposition {
  cudaq::quake::Pauli outer;
  cudaq::quake::Pauli middle;
  std::array<double, 3> angles;
  double phase;

  /// Decompose `u` into rotations about `outerAxis`, `middleAxis`, and
  /// `outerAxis`, which must be ZXZ or XZX.
  ///
  /// Rounds angles within rounding error of a multiple of pi/4.
  static EulerDecomposition get(const Eigen::Matrix2cd &u,
                                cudaq::quake::Pauli outerAxis,
                                cudaq::quake::Pauli middleAxis);

  /// Decompose `u3` into rotations.
  ///
  /// Decomposes into ZXZ or XZX based on the value of the `euler` attribute,
  /// if present. Defaults to ZXZ decomposition.
  static std::optional<EulerDecomposition> get(cudaq::quake::U3Op u3);

  /// The number of nonzero rotations `emit` creates.
  unsigned numRotations() const;

  /// The `u3(theta, phi, lambda)` angles of the represented matrix, up to a
  /// global phase. Only valid for the ZXZ form.
  ///
  /// Rounds angles within rounding error of a multiple of pi/4.
  std::array<double, 3> toU3Angles() const;

  /// The name of this form for `eulerFormAttrName`.
  llvm::StringRef formName() const;

  /// Emit the nonzero rotations of the decomposition on the `input` wire.
  ///
  /// Returns the wire it produces.
  mlir::Value emit(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value input) const;
};

/// Parse an Euler form name. If valid returns the (outer, middle) axes.
std::optional<std::pair<cudaq::quake::Pauli, cudaq::quake::Pauli>>
parseEulerForm(llvm::StringRef name);

/// Snap `angle` to an exact multiple of pi/4 when it is within rounding error.
double snapAngleToQuarterTurn(double angle);

} // namespace cudaq::opt
