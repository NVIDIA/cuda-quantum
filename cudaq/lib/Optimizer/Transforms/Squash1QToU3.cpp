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
#include "SingleQubitEuler.h"
#include "common/EigenDense.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/OptimizationLevel.h"
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
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_SQUASH1QTOU3
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;
using cudaq::opt::EulerDecomposition;
using cudaq::quake::Pauli;

namespace {

/// What the pass minimizes.
enum class CostModel {
  /// The number of gates, counting each nonzero rotation of a decomposed `u3`.
  rotationCount,
  /// Clifford+T gates and rotations are free: the number of rotations that
  /// need approximate Clifford+T synthesis, then the number of rotations.
  cliffordTFree,
};

/// The cost of a one-qubit gate sequence, compared lexicographically. Under
/// `rotationCount`, `approximateRotations` is always zero.
struct Cost {
  unsigned approximateRotations = 0;
  unsigned totalRotations = 0;

  auto operator<=>(const Cost &) const = default;
};

/// Computes the costs for one run of the pass.
struct CostFunction {
  CostModel model;
  double epsilon;

  /// Count one rotation by `angle` unless it is zero.
  void addRotation(Cost &cost, double angle,
                   cudaq::opt::detail::clifford_t_rotation_kind kind) const {
    if (angle == 0.0)
      return;
    ++cost.totalRotations;
    if (model == CostModel::cliffordTFree)
      cost.approximateRotations += !cudaq::opt::detail::classifyCliffordTAngle(
          llvm::APFloat(angle), epsilon, kind);
  }

  /// Cost of the nonzero rotations that `decomposition` would emit.
  Cost get(const EulerDecomposition &decomposition) const {
    Cost cost;
    for (double angle : decomposition.angles)
      addRotation(cost, angle,
                  cudaq::opt::detail::clifford_t_rotation_kind::axis);
    return cost;
  }
};

} // namespace

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
           cudaq::quake::SOp, cudaq::quake::TOp, cudaq::quake::U3Op,
           cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(operation))
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

/// The form that `U3ToRotations` gives `u3`: the one its `euler` attribute
/// names, or ZXZ.
static EulerDecomposition getRecordedDecomposition(cudaq::quake::U3Op u3) {
  std::pair<Pauli, Pauli> axes{Pauli::Z, Pauli::X};
  if (auto name = u3->getAttrOfType<StringAttr>(cudaq::opt::eulerFormAttrName))
    if (auto parsed = cudaq::opt::parseEulerForm(name.getValue()))
      axes = *parsed;
  return EulerDecomposition::get(getMatrix(u3), axes.first, axes.second);
}

/// Cost of the gates already in a sequence accepted by `isFusable`. A `u3`
/// costs the rotations it decomposes into. Under `rotationCount`, every other
/// gate but `quake.phase` costs one; under `cliffordTFree`, only rotations
/// cost.
static Cost getChainCost(ArrayRef<Operation *> operations,
                         const CostFunction &costFunction) {
  Cost cost;
  for (Operation *operation : operations) {
    if (auto u3 = dyn_cast<cudaq::quake::U3Op>(operation)) {
      Cost u3Cost = costFunction.get(getRecordedDecomposition(u3));
      cost.approximateRotations += u3Cost.approximateRotations;
      cost.totalRotations += u3Cost.totalRotations;
      continue;
    }
    if (isa<cudaq::quake::PhaseOp>(operation))
      continue;
    if (costFunction.model == CostModel::rotationCount) {
      ++cost.totalRotations;
      continue;
    }
    if (!isa<cudaq::quake::R1Op, cudaq::quake::RxOp, cudaq::quake::RyOp,
             cudaq::quake::RzOp>(operation))
      continue;
    FloatAttr angle = getConstantAngle(
        cast<cudaq::quake::OperatorInterface>(operation).getParameters()[0]);
    costFunction.addRotation(
        cost, angle.getValue().convertToDouble(),
        isa<cudaq::quake::R1Op>(operation)
            ? cudaq::opt::detail::clifford_t_rotation_kind::r1
            : cudaq::opt::detail::clifford_t_rotation_kind::axis);
  }
  return cost;
}

/// The squash of a chain: its matrix, its ZXZ form, which gives the `u3`
/// angles, and the form `U3ToRotations` should emit.
struct Squash {
  Eigen::Matrix2cd matrix;
  EulerDecomposition zxz;
  EulerDecomposition preferred;
};

/// Return the squash of `operations`, a sequence accepted by `isFusable`, or
/// nothing when the chain should stay as it is.
///
/// The preferred form is the cheaper of ZXZ and XZX, with ties going to ZXZ.
/// The two middle axes for one outer axis differ only by a pi/2 shift of the
/// outer angles, which leaves the count of approximate rotations unchanged, so
/// ZYZ and XYX add nothing to ZXZ and XZX. A Y outer axis can save a rotation
/// locally, but Ry commutes with neither the control nor the target of a
/// controlled X, so it blocks the rotation merging that quake-simplify does
/// afterwards and raised the T count of Fermi-Hubbard circuits.
///
/// Unless `alwaysSquash` is set, the chain is squashed only when the preferred
/// form is strictly cheaper than the chain. Under `cliffordTFree`, a chain
/// without an approximate rotation or with a single rotation cannot get
/// strictly cheaper.
static std::optional<Squash> findSquash(ArrayRef<Operation *> operations,
                                        const CostFunction &costFunction,
                                        bool alwaysSquash) {
  if (llvm::all_of(operations, [](Operation *operation) {
        return isa<cudaq::quake::PhaseOp>(operation);
      }))
    return std::nullopt;
  const Cost chainCost = getChainCost(operations, costFunction);
  if (!alwaysSquash && costFunction.model == CostModel::cliffordTFree &&
      (chainCost.approximateRotations == 0 || chainCost.totalRotations == 1))
    return std::nullopt;

  Eigen::Matrix2cd product = Eigen::Matrix2cd::Identity();
  for (Operation *operation : operations)
    product = getMatrix(operation) * product;

  EulerDecomposition zxz = EulerDecomposition::get(product, Pauli::Z, Pauli::X);
  EulerDecomposition xzx = EulerDecomposition::get(product, Pauli::X, Pauli::Z);
  const Cost zxzCost = costFunction.get(zxz);
  const Cost xzxCost = costFunction.get(xzx);
  const bool preferXZX = xzxCost < zxzCost;
  const Cost bestCost = preferXZX ? xzxCost : zxzCost;
  if (!alwaysSquash && !(bestCost < chainCost))
    return std::nullopt;
  return Squash{product, zxz, preferXZX ? xzx : zxz};
}

/// Record the cheaper form on a `u3` that is a chain on its own, and leave it
/// otherwise unchanged.
static void recordPreferredForm(cudaq::quake::U3Op u3,
                                const CostFunction &costFunction) {
  const Eigen::Matrix2cd matrix = getMatrix(u3);
  EulerDecomposition zxz = EulerDecomposition::get(matrix, Pauli::Z, Pauli::X);
  EulerDecomposition xzx = EulerDecomposition::get(matrix, Pauli::X, Pauli::Z);
  if (costFunction.get(xzx) < costFunction.get(zxz))
    u3->setAttr(cudaq::opt::eulerFormAttrName,
                StringAttr::get(u3.getContext(), xzx.formName()));
  else
    u3->removeAttr(cudaq::opt::eulerFormAttrName);
}

/// Emit `squash` as one `u3` and a phase correction on `input`, and return the
/// wire it produces.
///
/// `U3ToRotations` later emits its own phase correction `p`, computed from the
/// matrix of this `u3`, and `canonicalize` adds the two. The correction here
/// is therefore the total phase, snapped to a multiple of pi/4 within
/// roundoff, minus `p`. When the total is zero, the two cancel exactly.
static Value emitSquash(OpBuilder &builder, Location location, Value input,
                        const Squash &squash) {
  std::array<double, 3> angles = squash.zxz.toU3Angles();
  llvm::SmallVector<Value, 3> parameters;
  for (double angle : angles)
    parameters.push_back(
        cudaq::opt::factory::createF64Constant(location, builder, angle));
  llvm::SmallVector<Value> targets{input};
  auto u3 = cudaq::quake::createAndThreadGate<cudaq::quake::U3Op>(
      builder, location, /*isAdj=*/UnitAttr{}, parameters, {}, targets);
  if (squash.preferred.outer != Pauli::Z)
    u3->setAttr(cudaq::opt::eulerFormAttrName,
                builder.getStringAttr(squash.preferred.formName()));
  const Eigen::Matrix2cd u3Matrix = getMatrix(u3);
  const EulerDecomposition expansion = EulerDecomposition::get(
      u3Matrix, squash.preferred.outer, squash.preferred.middle);
  // `squash.matrix` is `exp(i*c) * u3Matrix`, and the expansion contributes
  // `expansion.phase` on top of `c`.
  const double correction =
      std::arg((squash.matrix * u3Matrix.adjoint()).trace());
  const double total =
      cudaq::opt::snapAngleToQuarterTurn(correction + expansion.phase);
  Value phaseValue = cudaq::opt::factory::createF64Constant(
      location, builder, total - expansion.phase);
  return cudaq::opt::emitPhaseCorrection(builder, location, phaseValue, {}, {},
                                         targets.front())
      .anchor;
}

static void squashBlock(Block &block, const CostFunction &costFunction,
                        bool alwaysSquash) {
  auto chains = cudaq::opt::collectSingleQubitChains(block, isFusable);
  for (cudaq::opt::SingleQubitChain &chain : llvm::reverse(chains)) {
    auto gates = llvm::make_filter_range(chain.operations, [](Operation *op) {
      return !isa<cudaq::quake::PhaseOp>(op);
    });
    // A lone `u3` is already squashed. Only choose its form.
    if (llvm::hasSingleElement(gates))
      if (auto u3 = dyn_cast<cudaq::quake::U3Op>(*gates.begin())) {
        recordPreferredForm(u3, costFunction);
        continue;
      }
    std::optional<Squash> squash =
        findSquash(chain.operations, costFunction, alwaysSquash);
    if (!squash)
      continue;
    Operation *first = chain.operations.front();
    OpBuilder builder(first);
    Value output =
        emitSquash(builder, first->getLoc(), chain.inputs.front(), *squash);
    cudaq::opt::replaceSingleQubitChain(std::move(chain), {output});
  }
}

namespace {

class Squash1QToU3Pass
    : public cudaq::opt::impl::Squash1QToU3Base<Squash1QToU3Pass> {
public:
  using Squash1QToU3Base::Squash1QToU3Base;

  LogicalResult initialize(MLIRContext *context) override {
    if (cost == "rotation-count")
      model = CostModel::rotationCount;
    else if (cost == "clifford-t-free")
      model = CostModel::cliffordTFree;
    else
      return emitError(UnknownLoc::get(context),
                       "squash-1q-to-u3 cost must be rotation-count or "
                       "clifford-t-free");
    if (std::isfinite(epsilon) && epsilon >= 0.0)
      return success();
    return emitError(UnknownLoc::get(context),
                     "squash-1q-to-u3 epsilon must be finite and non-negative");
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (cudaq::runtime::isQuantumOptimizationDisabled(function))
      return;
    const CostFunction costFunction{model, epsilon};
    cudaq::opt::forEachNestedBlock(function.getBody(), [&](Block &block) {
      squashBlock(block, costFunction, alwaysSquash);
    });
  }

private:
  CostModel model = CostModel::rotationCount;
};

} // namespace
