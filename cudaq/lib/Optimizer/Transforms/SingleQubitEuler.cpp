/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SingleQubitEuler.h"
#include "CliffordTAngle.h"
#include "PhaseUtilities.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

using namespace mlir;
using cudaq::quake::Pauli;

/// One unit in the last place of a double in [4, 8). Every Euler angle has a
/// magnitude of at most 2*pi, which lies in that range, so the unit bounds the
/// spacing of doubles near any of them.
static constexpr double ulpAtTwoPi = 4 * std::numeric_limits<double>::epsilon();

/// The absolute drift, in multiples of `ulpAtTwoPi`, assumed for an Euler angle
/// computed from a product of constant gates. The bound covers the chains in
/// the tests and benchmarks; a chain that drifts further only misses a snap and
/// keeps an approximate rotation, and the rewrite stays exact up to the drift.
static constexpr double eulerAngleUlps = 8;

/// An Euler angle within this distance, about 7e-15 radians, of a degenerate
/// middle angle or a multiple of pi/4 is treated as that value.
/// https://github.com/NVIDIA/cuda-quantum/issues/5680 tracks making exact-angle
/// matching consistent across passes.
static constexpr double roundoffTolerance = eulerAngleUlps * ulpAtTwoPi;

double cudaq::opt::snapAngleToQuarterTurn(double angle) {
  if (auto quarterTurns = cudaq::opt::detail::matchMultipleOfPi(
          llvm::APFloat(angle), /*denominator=*/4, roundoffTolerance))
    return cudaq::opt::detail::canonicalMultipleOfPi(
               *quarterTurns, /*denominator=*/4, llvm::APFloat::IEEEdouble())
        .convertToDouble();
  return angle;
}

/// Decompose a unitary into rotations about `outer`, `middle`, `outer`, where
/// the basis is ZXZ or XZX. This is the Euler decomposition of a one-qubit
/// unitary (Barenco et al., https://arxiv.org/abs/quant-ph/9503016, Lemma 4.1)
/// with the axes relabeled.
///
/// Write the special unitary as `w*I - i*(x*X + y*Y + z*Z)`. With `a` the
/// outer axis, `b` the middle axis, and `c` the remaining axis, the middle
/// angle is `2*atan2(hypot(b, c), hypot(w, a))` and the outer angles are
/// `atan2(a, w) +/- atan2(+/-c, b)`, where the inner sign is negative exactly
/// when (a, b, c) is a cyclic permutation of (x, y, z). In both bases the
/// remaining axis is Y, and only ZXZ, as (z, x, y), is cyclic. Every angle
/// comes from an `atan2`, so the result is defined for singular unitaries and
/// unchanged by a uniform scaling of the input, which absorbs roundoff drift
/// from long products.
cudaq::opt::EulerDecomposition
cudaq::opt::EulerDecomposition::get(const Eigen::Matrix2cd &u, Pauli outerAxis,
                                    Pauli middleAxis) {
  // The coefficients below are indexed by `Pauli`.
  static_assert(static_cast<unsigned>(Pauli::I) == 0 &&
                static_cast<unsigned>(Pauli::X) == 1 &&
                static_cast<unsigned>(Pauli::Y) == 2 &&
                static_cast<unsigned>(Pauli::Z) == 3);
  assert(((outerAxis == Pauli::Z && middleAxis == Pauli::X) ||
          (outerAxis == Pauli::X && middleAxis == Pauli::Z)) &&
         "only ZXZ and XZX are emitted");
  using namespace std::complex_literals;
  const double globalPhase = std::arg(u.determinant()) / 2;
  const Eigen::Matrix2cd su = std::exp(-1i * globalPhase) * u;

  const std::array<double, 4> coefficients{
      (su(0, 0).real() + su(1, 1).real()) / 2,
      -(su(0, 1).imag() + su(1, 0).imag()) / 2,
      (su(1, 0).real() - su(0, 1).real()) / 2,
      (su(1, 1).imag() - su(0, 0).imag()) / 2};
  const double w = coefficients[static_cast<unsigned>(Pauli::I)];
  const double a = coefficients[static_cast<unsigned>(outerAxis)];
  const double b = coefficients[static_cast<unsigned>(middleAxis)];
  const double c = coefficients[static_cast<unsigned>(Pauli::Y)];
  const bool cyclic = outerAxis == Pauli::Z;

  const double sum = std::atan2(a, w);
  const double difference = std::atan2(cyclic ? -c : c, b);
  const double middleAngle = 2 * std::atan2(std::hypot(b, c), std::hypot(w, a));
  std::array<double, 3> eulerAngles{sum + difference, middleAngle,
                                    sum - difference};
  // The outer angles are only determined as a pair when the middle rotation
  // is neither the identity nor a half turn. Otherwise their split is rounding
  // noise, so use the sum or difference that is well defined and put it in
  // one rotation.
  if (middleAngle <= roundoffTolerance)
    eulerAngles = {2 * sum, 0.0, 0.0};
  else if (std::numbers::pi - middleAngle <= roundoffTolerance)
    eulerAngles = {2 * difference, std::numbers::pi, 0.0};
  // A rotation that is a multiple of pi/4 in exact arithmetic comes out a few
  // ulps off. Replace it with the canonical value so that it counts as a
  // quarter turn even with a zero tolerance, and so that a zero rotation
  // neither counts as a rotation nor survives as a no-op.
  for (double &angle : eulerAngles)
    angle = snapAngleToQuarterTurn(angle);
  return {outerAxis, middleAxis, eulerAngles, globalPhase};
}

unsigned cudaq::opt::EulerDecomposition::numRotations() const {
  return llvm::count_if(angles, [](double angle) { return angle != 0.0; });
}

llvm::StringRef cudaq::opt::EulerDecomposition::formName() const {
  return outer == Pauli::Z ? "zxz" : "xzx";
}

std::optional<std::pair<Pauli, Pauli>>
cudaq::opt::parseEulerForm(llvm::StringRef name) {
  if (name == "zxz")
    return std::pair{Pauli::Z, Pauli::X};
  if (name == "xzx")
    return std::pair{Pauli::X, Pauli::Z};
  return std::nullopt;
}

/// `U3(theta, phi, lambda) = exp(i(phi+lambda)/2) Rz(phi) Ry(theta) Rz(lambda)`
/// and `Ry(theta) = Rz(pi/2) Rx(theta) Rz(-pi/2)`, so the ZXZ form
/// `Rz(a2) Rx(b) Rz(a0)` is `U3(b, a2 - pi/2, a0 + pi/2)` up to the phase
/// `(a0 + a2)/2`.
std::array<double, 3> cudaq::opt::EulerDecomposition::toU3Angles() const {
  assert(outer == Pauli::Z && middle == Pauli::X &&
         "u3 angles are read from the ZXZ form");
  constexpr double halfPi = std::numbers::pi / 2;
  const double theta = angles[1];
  double phi = angles[2] - halfPi;
  double lambda = angles[0] + halfPi;
  // Without a middle rotation only phi + lambda is meaningful. Put it all in
  // lambda, which reads as `U3(0, 0, lambda) = R1(lambda)`.
  if (theta == 0.0) {
    lambda = angles[0] + angles[2];
    phi = 0.0;
  }
  return {theta, snapAngleToQuarterTurn(phi), snapAngleToQuarterTurn(lambda)};
}

template <typename OpTy>
static Value emitRotation(OpBuilder &builder, Location location, Value wire,
                          double angle) {
  Value parameter =
      cudaq::opt::factory::createF64Constant(location, builder, angle);
  llvm::SmallVector<Value> targets{wire};
  cudaq::quake::createAndThreadGate<OpTy>(builder, location,
                                          /*isAdj=*/UnitAttr{},
                                          ValueRange{parameter}, {}, targets);
  return targets.front();
}

Value cudaq::opt::EulerDecomposition::emit(OpBuilder &builder,
                                           Location location,
                                           Value input) const {
  const std::array<Pauli, 3> axes{outer, middle, outer};
  Value wire = input;
  for (auto [rotationAxis, angle] : llvm::zip(axes, angles)) {
    if (angle == 0.0)
      continue;
    assert((rotationAxis == Pauli::X || rotationAxis == Pauli::Z) &&
           "only ZXZ and XZX are emitted");
    wire =
        rotationAxis == Pauli::X
            ? emitRotation<cudaq::quake::RxOp>(builder, location, wire, angle)
            : emitRotation<cudaq::quake::RzOp>(builder, location, wire, angle);
  }
  Value phaseValue =
      cudaq::opt::factory::createF64Constant(location, builder, phase);
  return cudaq::opt::emitPhaseCorrection(builder, location, phaseValue, {}, {},
                                         wire)
      .anchor;
}
