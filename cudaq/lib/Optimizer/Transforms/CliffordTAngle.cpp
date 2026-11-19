/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CliffordTAngle.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

// This file does not define a pass. It holds the angle classification shared by
// the passes that snap rotations to Clifford+T quarter turns.

// Every IEEE angle type up to f128, and x87 extended precision, converts to
// IEEE quad exactly. Quad's guard bits keep the canonical value from being
// built on an already-rounded pi. `matchMultipleOfPi` rejects a type, such as
// PowerPC double-double, whose values do not all convert exactly.
static llvm::APFloat quadPi() {
  return llvm::APFloat(
      llvm::APFloat::IEEEquad(),
      "3.14159265358979323846264338327950288419716939937510582097494");
}

static llvm::APFloat toQuad(std::int64_t value) {
  llvm::APFloat result = llvm::APFloat::getZero(llvm::APFloat::IEEEquad());
  result.convertFromAPInt(llvm::APInt(64, static_cast<std::uint64_t>(value),
                                      /*isSigned=*/true),
                          /*IsSigned=*/true,
                          llvm::APFloat::rmNearestTiesToEven);
  return result;
}

// The product and quotient each round in quad before the final rounding to
// `semantics`. With a power-of-two denominator the division is exact, so k*pi/4
// rounds exactly as a single rounding would.
llvm::APFloat
cudaq::opt::detail::canonicalMultipleOfPi(std::int64_t k,
                                          std::int64_t denominator,
                                          const llvm::fltSemantics &semantics) {
  assert(denominator > 0 && "denominator must be positive");
  llvm::APFloat value = quadPi();
  value.multiply(toQuad(k), llvm::APFloat::rmNearestTiesToEven);
  value.divide(toQuad(denominator), llvm::APFloat::rmNearestTiesToEven);
  bool losesInfo = false;
  value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value;
}

std::optional<std::int64_t> cudaq::opt::detail::matchMultipleOfPi(
    const llvm::APFloat &angle, std::int64_t denominator, double roundoff) {
  assert(denominator > 0 && "denominator must be positive");
  if (!angle.isFinite())
    return std::nullopt;
  llvm::APFloat quadAngle = angle;
  bool losesInfo = false;
  quadAngle.convert(llvm::APFloat::IEEEquad(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo)
    return std::nullopt;

  llvm::APFloat multiple = quadAngle;
  multiple.multiply(toQuad(denominator), llvm::APFloat::rmNearestTiesToEven);
  multiple.divide(quadPi(), llvm::APFloat::rmNearestTiesToEven);
  llvm::APSInt k(64, /*isUnsigned=*/false);
  bool isExact = false;
  if (multiple.convertToInteger(k, llvm::APFloat::rmNearestTiesToEven,
                                &isExact) &
      llvm::APFloat::opInvalidOp)
    return std::nullopt;

  llvm::APFloat distance =
      canonicalMultipleOfPi(k.getExtValue(), denominator, angle.getSemantics());
  distance.convert(llvm::APFloat::IEEEquad(),
                   llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  distance.subtract(quadAngle, llvm::APFloat::rmNearestTiesToEven);
  distance.clearSign();
  llvm::APFloat bound(roundoff);
  bound.convert(llvm::APFloat::IEEEquad(), llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
  if (distance.compare(bound) == llvm::APFloat::cmpGreaterThan)
    return std::nullopt;
  return k.getExtValue();
}

// The error, in units in the last place, assumed for each angle `addAngles`
// adds. Each angle is taken as the rounded value of some real angle: an input
// constant is within half a unit of it, and an Euler angle from
// fuse-1q-chains-for-clifford-t is within a few. An angle that has absorbed
// several earlier merges can drift further; its sum then keeps the computed
// value.
static constexpr double addendUlps = 4;

// Bound, in radians, on the error of `lhs + rhs` when each operand is within
// `addendUlps` units in the last place of its real angle. The sum rounds once
// more, by at most half a unit. The units are measured against the operands'
// magnitudes rather than the sum's, so the bound holds when they cancel, and it
// grows with them: for operands near 1e6 it is about 2e-9 radians. It assumes
// normal operands and is infinite when their magnitudes overflow a double.
static double sumRoundoff(const llvm::APFloat &lhs, const llvm::APFloat &rhs) {
  const llvm::fltSemantics &semantics = lhs.getSemantics();
  // One unit in the last place of `x` is at most `machineEpsilon * |x|`.
  const double machineEpsilon = std::ldexp(
      1.0, 1 - static_cast<int>(llvm::APFloat::semanticsPrecision(semantics)));
  double magnitude = 0.0;
  for (llvm::APFloat operand : {lhs, rhs}) {
    operand.clearSign();
    bool losesInfo = false;
    operand.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloat::rmTowardPositive, &losesInfo);
    magnitude += operand.convertToDouble();
  }
  return (addendUlps + 0.5) * machineEpsilon * magnitude;
}

llvm::APFloat cudaq::opt::detail::addAngles(const llvm::APFloat &lhs,
                                            const llvm::APFloat &rhs) {
  assert(&lhs.getSemantics() == &rhs.getSemantics() &&
         "angles share one floating-point type");
  llvm::APFloat sum = lhs;
  sum.add(rhs, llvm::APFloat::rmNearestTiesToEven);
  if (auto quarterTurns =
          matchMultipleOfPi(sum, /*denominator=*/4, sumRoundoff(lhs, rhs)))
    return canonicalMultipleOfPi(*quarterTurns, /*denominator=*/4,
                                 sum.getSemantics());
  return sum;
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
  const llvm::APFloat canonical = canonicalMultipleOfPi(
      quarterTurns, /*denominator=*/4, angle.getSemantics());
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
