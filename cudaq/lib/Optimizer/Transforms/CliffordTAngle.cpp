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

// IEEE quad provides guard bits for the multiplication, avoiding classification
// based on an already-rounded value of pi/4.
llvm::APFloat
cudaq::opt::detail::canonicalQuarterTurns(int64_t quarterTurns,
                                          const llvm::fltSemantics &semantics) {
  llvm::APFloat value(
      llvm::APFloat::IEEEquad(),
      "0.785398163397448309615660845819875721049292349843776455243736");
  llvm::APFloat factor = llvm::APFloat::getZero(llvm::APFloat::IEEEquad());
  factor.convertFromAPInt(
      llvm::APInt(64, static_cast<uint64_t>(quarterTurns), true), true,
      llvm::APFloat::rmNearestTiesToEven);
  value.multiply(factor, llvm::APFloat::rmNearestTiesToEven);
  bool losesInfo = false;
  value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value;
}

std::optional<int64_t> cudaq::opt::detail::classifyCliffordTAngle(
    const llvm::APFloat &angle, double epsilon, clifford_t_rotation_kind kind) {
  constexpr double quarterTurn = llvm::numbers::pi / 4.0;
  const double angleAsDouble = angle.convertToDouble();
  if (!std::isfinite(angleAsDouble))
    return std::nullopt;

  const double scaled = angleAsDouble / quarterTurn;
  constexpr double signedInt64Limit = 0x1p63;
  if (!std::isfinite(scaled) || scaled < -signedInt64Limit ||
      scaled >= signedInt64Limit)
    return std::nullopt;

  const double nearest = std::nearbyint(scaled);
  const auto quarterTurns = static_cast<int64_t>(nearest);
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
    const double angleDivisor =
        kind == clifford_t_rotation_kind::r1 ? 2.0 : 4.0;
    const double distance = 2.0 * std::sin(delta / angleDivisor);
    if (!std::isfinite(distance) || distance > epsilon)
      return std::nullopt;
  }

  return quarterTurns;
}
