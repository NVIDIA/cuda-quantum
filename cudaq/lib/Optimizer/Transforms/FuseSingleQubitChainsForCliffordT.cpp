/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CliffordTAngle.h"
#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "SingleQubitChain.h"
#include "common/EigenDense.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <array>
#include <cmath>
#include <compare>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_FUSESINGLEQUBITCHAINSFORCLIFFORDT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// The represented matrix is
/// `exp(i*phase) R_outer(angles[2]) R_middle(angles[1]) R_outer(angles[0])`.
struct EulerDecomposition {
  cudaq::quake::Pauli outer;
  cudaq::quake::Pauli middle;
  std::array<double, 3> angles;
  double phase;

  /// Decompose `u` into rotations about `outerAxis`, `middleAxis`, and
  /// `outerAxis`, which must be ZXZ or XZX.
  static EulerDecomposition get(const Eigen::Matrix2cd &u,
                                cudaq::quake::Pauli outerAxis,
                                cudaq::quake::Pauli middleAxis);

  /// Emit the decomposition on `input` and return the wire it produces.
  Value emit(OpBuilder &builder, Location location, Value input) const;
};

/// The rotations in a one-qubit gate sequence, compared by the number that
/// need approximate Clifford+T synthesis and then by the total number. The
/// first dominates because each approximation is synthesized separately.
struct RotationCost {
  unsigned approximateRotations = 0;
  unsigned totalRotations = 0;

  /// Cost of the rotations already in a sequence of operations accepted by
  /// `isFusable`.
  static RotationCost get(ArrayRef<Operation *> operations, double epsilon);
  /// Cost of the nonzero rotations that `decomposition` would emit.
  static RotationCost get(const EulerDecomposition &decomposition,
                          double epsilon);

  /// Count `angle` unless it is zero.
  void addRotation(const llvm::APFloat &angle, double epsilon,
                   cudaq::opt::detail::clifford_t_rotation_kind kind);

  auto operator<=>(const RotationCost &) const = default;
};

} // namespace

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

/// Return the value of `parameter` when it is a finite `arith.constant` that
/// converts to double exactly, as the angle classifier requires, and null
/// otherwise. `arith.constant` is the only producer `getOperatorMatrix` reads.
/// `factory::maybeValueOfFloatConstant` is not used because it accepts any
/// constant-like operation and does not check the precision.
static FloatAttr getConstantAngle(Value parameter) {
  auto constant = parameter.getDefiningOp<arith::ConstantOp>();
  auto angle = constant ? dyn_cast<FloatAttr>(constant.getValue()) : nullptr;
  if (!angle || !angle.getValue().isFinite() ||
      !llvm::APFloat::isRepresentableBy(angle.getValue().getSemantics(),
                                        llvm::APFloat::IEEEdouble()))
    return nullptr;
  return angle;
}

/// Return whether the pass composes `operation`: an uncontrolled single-qubit
/// gate whose parameters are all accepted by `getConstantAngle`. Any other
/// parameter ends the chain.
static bool isFusable(Operation *operation) {
  if (!isa<cudaq::quake::HOp, cudaq::quake::PhaseOp, cudaq::quake::R1Op,
           cudaq::quake::RxOp, cudaq::quake::RyOp, cudaq::quake::RzOp,
           cudaq::quake::SOp, cudaq::quake::TOp, cudaq::quake::XOp,
           cudaq::quake::YOp, cudaq::quake::ZOp>(operation))
    return false;
  auto gate = cast<cudaq::quake::OperatorInterface>(operation);
  return gate.getControls().empty() && gate.getTargets().size() == 1 &&
         llvm::all_of(gate.getParameters(), [](Value parameter) {
           return static_cast<bool>(getConstantAngle(parameter));
         });
}

/// Return the matrix of a gate accepted by `isFusable`.
static Eigen::Matrix2cd getMatrix(Operation *operation) {
  llvm::SmallVector<std::complex<double>, 4> columnMajor;
  cast<cudaq::quake::OperatorInterface>(operation).getOperatorMatrix(
      columnMajor);
  assert(columnMajor.size() == 4 && "fusable gates have constant parameters");
  return Eigen::Map<const Eigen::Matrix2cd>(columnMajor.data());
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
EulerDecomposition EulerDecomposition::get(const Eigen::Matrix2cd &u,
                                           cudaq::quake::Pauli outerAxis,
                                           cudaq::quake::Pauli middleAxis) {
  using cudaq::quake::Pauli;
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
  // ulps off. Replace it with the canonical value so that the classifier counts
  // it as a quarter turn even when epsilon is zero, and so that a zero rotation
  // neither counts as a rotation nor survives as a no-op.
  for (double &angle : eulerAngles)
    if (auto quarterTurns = cudaq::opt::detail::matchMultipleOfPi(
            llvm::APFloat(angle), /*denominator=*/4, roundoffTolerance))
      angle = cudaq::opt::detail::canonicalMultipleOfPi(
                  *quarterTurns, /*denominator=*/4, llvm::APFloat::IEEEdouble())
                  .convertToDouble();
  return {outerAxis, middleAxis, eulerAngles, globalPhase};
}

void RotationCost::addRotation(
    const llvm::APFloat &angle, double epsilon,
    cudaq::opt::detail::clifford_t_rotation_kind kind) {
  if (angle.isZero())
    return;
  ++totalRotations;
  approximateRotations +=
      !cudaq::opt::detail::classifyCliffordTAngle(angle, epsilon, kind);
}

RotationCost RotationCost::get(ArrayRef<Operation *> operations,
                               double epsilon) {
  RotationCost cost;
  for (Operation *operation : operations) {
    if (!isa<cudaq::quake::R1Op, cudaq::quake::RxOp, cudaq::quake::RyOp,
             cudaq::quake::RzOp>(operation))
      continue;
    FloatAttr angle = getConstantAngle(
        cast<cudaq::quake::OperatorInterface>(operation).getParameters()[0]);
    cost.addRotation(angle.getValue(), epsilon,
                     isa<cudaq::quake::R1Op>(operation)
                         ? cudaq::opt::detail::clifford_t_rotation_kind::r1
                         : cudaq::opt::detail::clifford_t_rotation_kind::axis);
  }
  return cost;
}

RotationCost RotationCost::get(const EulerDecomposition &decomposition,
                               double epsilon) {
  RotationCost cost;
  for (double angle : decomposition.angles)
    cost.addRotation(llvm::APFloat(angle), epsilon,
                     cudaq::opt::detail::clifford_t_rotation_kind::axis);
  return cost;
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

Value EulerDecomposition::emit(OpBuilder &builder, Location location,
                               Value input) const {
  using cudaq::quake::Pauli;
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

/// Return the decomposition that replaces `operations`, a sequence accepted by
/// `isFusable`, or nothing when no basis is strictly cheaper than the sequence.
///
/// Only a sequence with an approximate rotation is considered, since the rest
/// are already free. A sequence with a single rotation cannot get strictly
/// cheaper. The two middle axes for one outer axis differ only by a pi/2 shift
/// of the outer angles, which leaves the count of approximate rotations
/// unchanged, so ZYZ and XYX add nothing to ZXZ and XZX. A Y outer axis can
/// save a rotation locally, but Ry commutes with neither the control nor the
/// target of a controlled X, so it blocks the rotation merging that
/// quake-simplify does afterwards and raised the T count of Fermi-Hubbard
/// circuits. Ties keep the earlier basis.
static std::optional<EulerDecomposition>
findCheaperDecomposition(ArrayRef<Operation *> operations, double epsilon) {
  const RotationCost inputCost = RotationCost::get(operations, epsilon);
  if (inputCost.approximateRotations == 0 || inputCost.totalRotations == 1)
    return std::nullopt;

  Eigen::Matrix2cd product = Eigen::Matrix2cd::Identity();
  for (Operation *operation : operations)
    product = getMatrix(operation) * product;

  using cudaq::quake::Pauli;
  std::optional<EulerDecomposition> best;
  RotationCost bestCost = inputCost;
  for (auto [outer, middle] :
       {std::pair{Pauli::Z, Pauli::X}, std::pair{Pauli::X, Pauli::Z}}) {
    EulerDecomposition candidate =
        EulerDecomposition::get(product, outer, middle);
    if (RotationCost cost = RotationCost::get(candidate, epsilon);
        cost < bestCost) {
      best = candidate;
      bestCost = cost;
    }
  }
  return best;
}

static void fuseBlock(Block &block, double epsilon) {
  auto chains = cudaq::opt::collectSingleQubitChains(block, isFusable);
  for (cudaq::opt::SingleQubitChain &chain : llvm::reverse(chains)) {
    std::optional<EulerDecomposition> best =
        findCheaperDecomposition(chain.operations, epsilon);
    if (!best)
      continue;
    Operation *first = chain.operations.front();
    OpBuilder builder(first);
    Value output = best->emit(builder, first->getLoc(), chain.inputs.front());
    cudaq::opt::replaceSingleQubitChain(std::move(chain), {output});
  }
}

namespace {

class FuseSingleQubitChainsForCliffordTPass
    : public cudaq::opt::impl::FuseSingleQubitChainsForCliffordTBase<
          FuseSingleQubitChainsForCliffordTPass> {
public:
  using FuseSingleQubitChainsForCliffordTBase::
      FuseSingleQubitChainsForCliffordTBase;

  LogicalResult initialize(MLIRContext *context) override {
    if (std::isfinite(epsilon) && epsilon >= 0.0)
      return success();
    return emitError(UnknownLoc::get(context),
                     "fuse-1q-chains-for-clifford-t epsilon must be finite and "
                     "non-negative");
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function->hasAttr(cudaq::runtime::disableQuantumOpts))
      return;
    cudaq::opt::forEachNestedBlock(
        function.getBody(), [&](Block &block) { fuseBlock(block, epsilon); });
  }
};

} // namespace
