/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CliffordTAngle.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <cstdint>
#include <limits>

// This file does not define a pass. It holds the angle classification shared by
// the passes that snap rotations to Clifford+T quarter turns.

// IEEE quad provides guard bits for the multiplication, avoiding classification
// based on an already-rounded value of pi/4.
llvm::APFloat
cudaq::opt::detail::canonicalQuarterTurns(std::int64_t quarterTurns,
                                          const llvm::fltSemantics &semantics) {
  llvm::APFloat value(
      llvm::APFloat::IEEEquad(),
      "0.785398163397448309615660845819875721049292349843776455243736");
  llvm::APFloat factor = llvm::APFloat::getZero(llvm::APFloat::IEEEquad());
  factor.convertFromAPInt(
      llvm::APInt(64, static_cast<std::uint64_t>(quarterTurns), true), true,
      llvm::APFloat::rmNearestTiesToEven);
  value.multiply(factor, llvm::APFloat::rmNearestTiesToEven);
  bool losesInfo = false;
  value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value;
}

std::optional<std::int64_t> cudaq::opt::detail::classifyCliffordTAngle(
    const llvm::APFloat &angle, double epsilon, clifford_t_rotation_kind kind) {
  constexpr double quarterTurn = llvm::numbers::pi / 4.0;
  const double angleAsDouble = angle.convertToDouble();
  if (!std::isfinite(angleAsDouble))
    return std::nullopt;

  const double scaled = angleAsDouble / quarterTurn;
  // The quarter-turn count must fit in an int64. This is 2^63, exactly.
  constexpr double int64Bound =
      -static_cast<double>(std::numeric_limits<std::int64_t>::min());
  if (!std::isfinite(scaled) || scaled < -int64Bound || scaled >= int64Bound)
    return std::nullopt;

  const double nearest = std::nearbyint(scaled);
  const auto quarterTurns = static_cast<std::int64_t>(nearest);
  const llvm::APFloat canonical =
      canonicalQuarterTurns(quarterTurns, angle.getSemantics());
  if (epsilon == 0.0) {
    if (!angle.bitwiseIsEqual(canonical))
      return std::nullopt;
  } else {
    const double delta = std::abs(angleAsDouble - canonical.convertToDouble());
    // For `delta = abs(theta - canonical)`, the spectral-norm distance
    // `||U(theta) - U(canonical)||_2` is `2*sin(delta/2)` for
    // `R1(theta) = diag(1, exp(i*theta))` and `2*sin(delta/4)` for axis
    // rotations `U(theta) = exp(-i*theta*sigma/2)`.
    const bool isR1 = kind == clifford_t_rotation_kind::r1;
    const double distance = 2.0 * std::sin(delta / (isR1 ? 2.0 : 4.0));
    if (!std::isfinite(distance) || distance > epsilon)
      return std::nullopt;
  }

  return quarterTurns;
}
