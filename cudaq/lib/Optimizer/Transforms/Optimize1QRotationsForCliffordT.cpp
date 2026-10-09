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
#include "ScalarWireChain.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_OPTIMIZE1QROTATIONSFORCLIFFORDT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Row-major [u00, u01, u10, u11].
using Matrix = std::array<std::complex<double>, 4>;

enum class axis : unsigned { x, y, z };

/// The represented matrix is
/// `exp(i*phase) R_outer(angles[2]) R_middle(angles[1]) R_outer(angles[0])`.
struct EulerDecomposition {
  axis outer;
  axis middle;
  std::array<double, 3> angles;
  double phase;
};

/// What a rotation sequence costs downstream. Rotations that are not Clifford+T
/// angles each need approximate synthesis, so they dominate. The rotation count
/// breaks ties and keeps chains in their shortest form.
struct Cost {
  unsigned approximate = 0;
  unsigned rotations = 0;

  auto operator<=>(const Cost &) const = default;
};

class Optimize1QRotationsForCliffordTPass
    : public cudaq::opt::impl::Optimize1QRotationsForCliffordTBase<
          Optimize1QRotationsForCliffordTPass> {
public:
  using Optimize1QRotationsForCliffordTBase::
      Optimize1QRotationsForCliffordTBase;
  void runOnOperation() override;
};

} // namespace

/// Return the matrix of an uncontrolled single-qubit gate with constant
/// parameters, or nothing if the gate is not one the pass composes.
static std::optional<Matrix> getMatrix(Operation *operation) {
  if (!isa<cudaq::quake::HOp, cudaq::quake::PhaseOp, cudaq::quake::R1Op,
           cudaq::quake::RxOp, cudaq::quake::RyOp, cudaq::quake::RzOp,
           cudaq::quake::SOp, cudaq::quake::TOp, cudaq::quake::XOp,
           cudaq::quake::YOp, cudaq::quake::ZOp>(operation))
    return std::nullopt;
  auto gate = cast<cudaq::quake::OperatorInterface>(operation);
  if (!gate.getControls().empty() || gate.getTargets().size() != 1)
    return std::nullopt;

  // The matrix is column-major and empty unless every parameter is constant.
  llvm::SmallVector<std::complex<double>, 4> columnMajor;
  gate.getOperatorMatrix(columnMajor);
  if (columnMajor.size() != 4 ||
      !llvm::all_of(columnMajor, [](std::complex<double> value) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
      }))
    return std::nullopt;
  return Matrix{columnMajor[0], columnMajor[2], columnMajor[1], columnMajor[3]};
}

static Matrix multiply(const Matrix &lhs, const Matrix &rhs) {
  return {lhs[0] * rhs[0] + lhs[1] * rhs[2], lhs[0] * rhs[1] + lhs[1] * rhs[3],
          lhs[2] * rhs[0] + lhs[3] * rhs[2], lhs[2] * rhs[1] + lhs[3] * rhs[3]};
}

/// Decompose a unitary into rotations about `outer`, `middle`, `outer`.
///
/// Write the special unitary as `w*I - i*(x*X + y*Y + z*Z)`. With `a` the
/// outer axis, `b` the middle axis, and `c` the remaining axis, the middle
/// angle is `2*atan2(hypot(b, c), hypot(w, a))` and the outer angles are
/// `atan2(a, w) +/- atan2(+/-c, b)`, where the inner sign is negative exactly
/// when (a, b, c) is a cyclic permutation of (x, y, z). Every angle comes from
/// an `atan2`, so the result is defined for singular unitaries and unchanged
/// by a uniform scaling of the input, which absorbs roundoff drift from long
/// products.
static EulerDecomposition decompose(const Matrix &u, axis outerAxis,
                                    axis middleAxis) {
  const auto outer = static_cast<unsigned>(outerAxis);
  const auto middle = static_cast<unsigned>(middleAxis);
  using namespace std::complex_literals;
  const double phase = std::arg(u[0] * u[3] - u[1] * u[2]) / 2;
  const std::complex<double> unphase = std::exp(-1i * phase);
  const Matrix su{u[0] * unphase, u[1] * unphase, u[2] * unphase,
                  u[3] * unphase};

  const double w = (su[0].real() + su[3].real()) / 2;
  const std::array<double, 3> v{-(su[1].imag() + su[2].imag()) / 2,
                                (su[2].real() - su[1].real()) / 2,
                                (su[3].imag() - su[0].imag()) / 2};
  const unsigned rest = 3 - outer - middle;
  const bool cyclic = middle == (outer + 1) % 3;

  const double sum = std::atan2(v[outer], w);
  const double difference = std::atan2(cyclic ? -v[rest] : v[rest], v[middle]);
  const double middleAngle =
      2 * std::atan2(std::hypot(v[middle], v[rest]), std::hypot(w, v[outer]));
  constexpr double tolerance = 32 * std::numeric_limits<double>::epsilon();
  std::array<double, 3> angles{sum + difference, middleAngle, sum - difference};
  // The outer angles are only determined as a pair when the middle rotation
  // is neither the identity nor a half turn. Otherwise their split is rounding
  // noise, so use the sum or difference that is well defined and put it in
  // one rotation.
  if (middleAngle <= tolerance)
    angles = {2 * sum, 0.0, 0.0};
  else if (std::numbers::pi - middleAngle <= tolerance)
    angles = {2 * difference, std::numbers::pi, 0.0};
  // A rotation that is zero in exact arithmetic comes out as roundoff. Drop it
  // so that it neither counts as a rotation nor survives as a no-op.
  for (double &angle : angles)
    if (std::abs(angle) <= tolerance)
      angle = 0.0;
  return {outerAxis, middleAxis, angles, phase};
}

static void addRotation(Cost &cost, const llvm::APFloat &angle, double epsilon,
                        cudaq::opt::detail::clifford_t_rotation_kind kind) {
  if (angle.isZero())
    return;
  ++cost.rotations;
  cost.approximate +=
      !cudaq::opt::detail::classifyCliffordTAngle(angle, epsilon, kind);
}

/// Cost of the rotations already in a chain. The collector accepted only
/// operations whose parameters are constant.
static Cost getCost(ArrayRef<Operation *> operations, double epsilon) {
  Cost cost;
  for (Operation *operation : operations) {
    if (!isa<cudaq::quake::R1Op, cudaq::quake::RxOp, cudaq::quake::RyOp,
             cudaq::quake::RzOp>(operation))
      continue;
    FloatAttr angle;
    matchPattern(
        cast<cudaq::quake::OperatorInterface>(operation).getParameters()[0],
        m_Constant(&angle));
    addRotation(cost, angle.getValue(), epsilon,
                isa<cudaq::quake::R1Op>(operation)
                    ? cudaq::opt::detail::clifford_t_rotation_kind::r1
                    : cudaq::opt::detail::clifford_t_rotation_kind::axis);
  }
  return cost;
}

static Cost getCost(const EulerDecomposition &decomposition, double epsilon) {
  Cost cost;
  for (double angle : decomposition.angles)
    addRotation(cost, llvm::APFloat(angle), epsilon,
                cudaq::opt::detail::clifford_t_rotation_kind::axis);
  return cost;
}

template <typename OpTy>
static Value emitRotation(OpBuilder &builder, Location location, Value wire,
                          double angle) {
  Value parameter =
      cudaq::opt::factory::createF64Constant(location, builder, angle);
  llvm::SmallVector<Value> targets{wire};
  cudaq::quake::createAndThreadGate<OpTy>(builder, location, UnitAttr{},
                                          ValueRange{parameter}, {}, targets);
  return targets.front();
}

/// Emit `decomposition` on `input` and return the wire it produces.
static Value emit(OpBuilder &builder, Location location, Value input,
                  const EulerDecomposition &decomposition) {
  const std::array<axis, 3> axes{decomposition.outer, decomposition.middle,
                                 decomposition.outer};
  Value wire = input;
  for (auto [axis, angle] : llvm::zip(axes, decomposition.angles)) {
    if (angle == 0.0)
      continue;
    switch (axis) {
    case axis::x:
      wire = emitRotation<cudaq::quake::RxOp>(builder, location, wire, angle);
      break;
    case axis::y:
      wire = emitRotation<cudaq::quake::RyOp>(builder, location, wire, angle);
      break;
    case axis::z:
      wire = emitRotation<cudaq::quake::RzOp>(builder, location, wire, angle);
      break;
    }
  }
  Value phase = cudaq::opt::factory::createF64Constant(location, builder,
                                                       decomposition.phase);
  return cudaq::opt::emitPhaseCorrection(builder, location, phase, {}, {}, wire)
      .anchor;
}

static void optimizeBlock(Block &block, double epsilon) {
  auto chains = cudaq::opt::collectScalarWireChains(
      block,
      [](Operation *operation) { return getMatrix(operation).has_value(); },
      [](Operation *, Operation *) { return true; });

  // The collected chains borrow IR values, so rewrite them in reverse order.
  for (cudaq::opt::ScalarWireChain &chain : llvm::reverse(chains)) {
    const Cost inputCost = getCost(chain.operations, epsilon);
    if (inputCost.approximate == 0)
      continue;

    Matrix product{1.0, 0.0, 0.0, 1.0};
    for (Operation *operation : chain.operations)
      product = multiply(*getMatrix(operation), product);

    // Ties keep the earlier basis, and the chain is rewritten only when the
    // best decomposition is strictly cheaper than the chain itself.
    std::optional<EulerDecomposition> best;
    Cost bestCost = inputCost;
    for (auto [outer, middle] :
         {std::pair{axis::z, axis::x}, std::pair{axis::x, axis::z},
          std::pair{axis::z, axis::y}}) {
      EulerDecomposition candidate = decompose(product, outer, middle);
      if (Cost cost = getCost(candidate, epsilon); cost < bestCost) {
        best = candidate;
        bestCost = cost;
      }
    }
    if (!best)
      continue;

    Operation *first = chain.operations.front();
    OpBuilder builder(first);
    Value output = emit(builder, first->getLoc(), chain.inputs.front(), *best);
    cudaq::opt::replaceScalarWireChain(std::move(chain), {output});
  }
}

static void optimizeRegion(Region &region, double epsilon) {
  for (Block &block : region) {
    optimizeBlock(block, epsilon);
    for (Operation &operation : block)
      for (Region &nested : operation.getRegions())
        optimizeRegion(nested, epsilon);
  }
}

void Optimize1QRotationsForCliffordTPass::runOnOperation() {
  func::FuncOp function = getOperation();
  if (!std::isfinite(epsilon) || epsilon < 0.0) {
    function.emitError(
        "optimize-1q-rotations-for-clifford-t epsilon must be finite and "
        "non-negative");
    signalPassFailure();
    return;
  }
  if (function->hasAttr(cudaq::runtime::disableQuantumOpts))
    return;
  optimizeRegion(function.getBody(), epsilon);
}
