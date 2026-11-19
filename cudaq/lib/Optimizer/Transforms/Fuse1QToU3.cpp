/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CliffordTAngle.h"
#include "DecompositionPatterns.h"
#include "EulerDecomposition.h"
#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "SingleQubitChain.h"
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
#include <memory>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_FUSE1QTOU3
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;
using cudaq::opt::EulerDecomposition;
using cudaq::quake::Pauli;

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

/// Return whether some entry of `basis` covers `gate`, an uncontrolled gate
/// accepted by `isFusable`. As in `basis-conversion`, the adjoint flag is
/// ignored.
static bool isInBasis(Operation *gate,
                      ArrayRef<cudaq::detail::OperatorInfo> basis) {
  const cudaq::detail::OperatorInfo info(gate->getName().stripDialect());
  return llvm::any_of(basis, [&](const cudaq::detail::OperatorInfo &entry) {
    return entry.covers(info);
  });
}

namespace {

/// The cost of a one-qubit gate sequence, compared lexicographically.
struct Cost {
  /// What the cost model minimizes as the first priority.
  unsigned primary = 0;
  /// What the cost model minimizes as a tie break.
  unsigned secondary = 0;

  Cost &operator+=(const Cost &other) {
    primary += other.primary;
    secondary += other.secondary;
    return *this;
  }

  auto operator<=>(const Cost &) const = default;
};

/// What the pass minimizes.
class CostModel {
public:
  virtual ~CostModel() = default;

  /// Cost of `gate`, a gate accepted by `isFusable` other than `quake.phase`
  /// and `quake.u3`.
  virtual Cost getGateCost(Operation *gate) const = 0;

  /// Cost of a `u3` that `U3ToRotations` expands in the form `decomposition`.
  virtual Cost
  getDecompositionCost(const EulerDecomposition &decomposition) const = 0;

  /// Return whether a chain of cost `chainCost` is known to be optimal, i.e. no
  /// fusion of it can be strictly cheaper. Used to short-circuit the search for
  /// a cheaper fusion.
  virtual bool isOptimal(const Cost &chainCost) const { return false; }

  /// Sum of costs of the gates in `operations`.
  ///
  /// Gate costs:
  ///  - `quake.phase` cost is 0,
  ///  - `quake.u3` cost is obtained using `getDecompositionCost`,
  ///  - Other gates cost is obtained using `getGateCost`.
  Cost getChainCost(ArrayRef<Operation *> operations) const {
    Cost cost;
    for (Operation *operation : operations) {
      if (isa<cudaq::quake::PhaseOp>(operation))
        continue;
      if (auto u3 = dyn_cast<cudaq::quake::U3Op>(operation)) {
        std::optional<EulerDecomposition> decomposition =
            EulerDecomposition::get(u3);
        assert(decomposition && "fusable gates have constant parameters");
        cost += getDecompositionCost(*decomposition);
        continue;
      }
      cost += getGateCost(operation);
    }
    return cost;
  }
};

/// Cost function to minimize the (estimated) number of gates after basis
/// conversion.
///
/// Every non-`u3` gate costs one. A `u3` costs one when `u3` is a basis gate,
/// and otherwise costs the number of rotations `U3ToRotations` expands it into.
class GateCountModel final : public CostModel {
public:
  explicit GateCountModel(bool containsU3) : containsU3(containsU3) {}

  Cost getGateCost(Operation *) const override { return {1, 0}; }

  Cost
  getDecompositionCost(const EulerDecomposition &decomposition) const override {
    return {containsU3 ? 1u : decomposition.numRotations(), 0};
  }

private:
  /// Whether the basis is empty or contains an uncontrolled `u3`.
  bool containsU3;
};

/// Cost function to minimize the number of rotations that cannot be expressed
/// with a single Cliffrd+T gate.
///
/// Rotations with angles that are multiples of pi/8 are free. Every other
/// rotation costs one.
class CliffordTFreeModel final : public CostModel {
public:
  /// `epsilon` is the operator-norm distance within which a rotation counts as
  /// a quarter turn.
  explicit CliffordTFreeModel(double epsilon) : epsilon(epsilon) {}

  Cost getGateCost(Operation *gate) const override {
    if (!isa<cudaq::quake::R1Op, cudaq::quake::RxOp, cudaq::quake::RyOp,
             cudaq::quake::RzOp>(gate))
      return {};
    FloatAttr angle = getConstantAngle(
        cast<cudaq::quake::OperatorInterface>(gate).getParameters()[0]);
    return getRotationCost(
        angle.getValue().convertToDouble(),
        isa<cudaq::quake::R1Op>(gate)
            ? cudaq::opt::detail::clifford_t_rotation_kind::r1
            : cudaq::opt::detail::clifford_t_rotation_kind::axis);
  }

  Cost
  getDecompositionCost(const EulerDecomposition &decomposition) const override {
    Cost cost;
    for (double angle : decomposition.angles)
      cost += getRotationCost(
          angle, cudaq::opt::detail::clifford_t_rotation_kind::axis);
    return cost;
  }

  /// A chain without an approximate rotation or with a single rotation cannot
  /// get strictly cheaper.
  bool isOptimal(const Cost &chainCost) const override {
    return chainCost.primary == 0 || chainCost.secondary == 1;
  }

private:
  /// Cost of one rotation by `angle`, which is free when `angle` is zero.
  Cost
  getRotationCost(double angle,
                  cudaq::opt::detail::clifford_t_rotation_kind kind) const {
    if (angle == 0.0)
      return {};
    const bool approximate = !cudaq::opt::detail::classifyCliffordTAngle(
        llvm::APFloat(angle), epsilon, kind);
    return {approximate ? 1u : 0u, 1};
  }

  double epsilon;
};

/// The fusion of a chain: its matrix, its ZXZ form, which gives the `u3`
/// angles, and the form `U3ToRotations` should emit.
struct Fused1QChain {
  Eigen::Matrix2cd matrix;
  EulerDecomposition zxz;
  EulerDecomposition preferred;
};

} // namespace

/// Return the fusion of `operations`, a sequence accepted by `isFusable`, or
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
/// Unless `force` is set, the chain is fused only when the preferred form is
/// strictly cheaper than the chain.
static std::optional<Fused1QChain> findFusion(ArrayRef<Operation *> operations,
                                              const CostModel &costModel,
                                              bool force) {
  if (llvm::all_of(operations, [](Operation *operation) {
        return isa<cudaq::quake::PhaseOp>(operation);
      }))
    return std::nullopt;
  const Cost chainCost = costModel.getChainCost(operations);
  if (!force && costModel.isOptimal(chainCost))
    return std::nullopt;

  Eigen::Matrix2cd product = Eigen::Matrix2cd::Identity();
  for (Operation *operation : operations)
    product = getMatrix(operation) * product;

  EulerDecomposition zxz = EulerDecomposition::get(product, Pauli::Z, Pauli::X);
  EulerDecomposition xzx = EulerDecomposition::get(product, Pauli::X, Pauli::Z);
  const Cost zxzCost = costModel.getDecompositionCost(zxz);
  const Cost xzxCost = costModel.getDecompositionCost(xzx);
  const bool preferXZX = xzxCost < zxzCost;
  const Cost bestCost = preferXZX ? xzxCost : zxzCost;
  if (!force && !(bestCost < chainCost))
    return std::nullopt;
  return Fused1QChain{product, zxz, preferXZX ? xzx : zxz};
}

/// Record the cheaper form on a `u3` that is a chain on its own, and leave it
/// otherwise unchanged.
static void recordPreferredForm(cudaq::quake::U3Op u3,
                                const CostModel &costModel) {
  const Eigen::Matrix2cd matrix = getMatrix(u3);
  EulerDecomposition zxz = EulerDecomposition::get(matrix, Pauli::Z, Pauli::X);
  EulerDecomposition xzx = EulerDecomposition::get(matrix, Pauli::X, Pauli::Z);
  if (costModel.getDecompositionCost(xzx) < costModel.getDecompositionCost(zxz))
    u3->setAttr(cudaq::opt::eulerFormAttrName,
                StringAttr::get(u3.getContext(), xzx.formName()));
  else
    u3->removeAttr(cudaq::opt::eulerFormAttrName);
}

/// Emit `fusion` as one `u3` and a phase correction on `input`, and return the
/// wire it produces.
///
/// `U3ToRotations` later emits its own phase correction `p`, computed from the
/// matrix of this `u3`, and `canonicalize` adds the two. The correction here
/// is therefore the total phase, snapped to a multiple of pi/4 within
/// roundoff, minus `p`. When the total is zero, the two cancel exactly.
static Value emitFusion(OpBuilder &builder, Location location, Value input,
                        const Fused1QChain &fusion) {
  std::array<double, 3> angles = fusion.zxz.toU3Angles();
  llvm::SmallVector<Value, 3> parameters;
  for (double angle : angles)
    parameters.push_back(
        cudaq::opt::factory::createF64Constant(location, builder, angle));
  llvm::SmallVector<Value> targets{input};
  auto u3 = cudaq::quake::createAndThreadGate<cudaq::quake::U3Op>(
      builder, location, /*isAdj=*/UnitAttr{}, parameters, {}, targets);
  if (fusion.preferred.outer != Pauli::Z)
    u3->setAttr(cudaq::opt::eulerFormAttrName,
                builder.getStringAttr(fusion.preferred.formName()));
  const Eigen::Matrix2cd u3Matrix = getMatrix(u3);
  const EulerDecomposition expansion = EulerDecomposition::get(
      u3Matrix, fusion.preferred.outer, fusion.preferred.middle);
  // `fusion.matrix` is `exp(i*c) * u3Matrix`, and the expansion contributes
  // `expansion.phase` on top of `c`.
  const double correction =
      std::arg((fusion.matrix * u3Matrix.adjoint()).trace());
  const double total =
      cudaq::opt::snapAngleToQuarterTurn(correction + expansion.phase);
  Value phaseValue = cudaq::opt::factory::createF64Constant(
      location, builder, total - expansion.phase);
  return cudaq::opt::emitPhaseCorrection(builder, location, phaseValue, {}, {},
                                         targets.front())
      .anchor;
}

/// Fuse the chains in `block`. When all gates in the chain are in `basis`, the
/// chain is fused only when the fusion is cheaper. If a gate is not in `basis,
/// the chain is always fused.
static void fuseBlock(Block &block, const CostModel &costModel,
                      ArrayRef<cudaq::detail::OperatorInfo> basis) {
  auto chains = cudaq::opt::collectSingleQubitChains(block, isFusable);
  for (cudaq::opt::SingleQubitChain &chain : llvm::reverse(chains)) {
    auto gates = llvm::make_filter_range(chain.operations, [](Operation *op) {
      return !isa<cudaq::quake::PhaseOp>(op);
    });
    // A lone `u3` is already fused. Only choose its form.
    if (llvm::hasSingleElement(gates))
      if (auto u3 = dyn_cast<cudaq::quake::U3Op>(*gates.begin())) {
        recordPreferredForm(u3, costModel);
        continue;
      }
    const bool force =
        !basis.empty() && llvm::any_of(gates, [&](Operation *gate) {
          return !isInBasis(gate, basis);
        });
    std::optional<Fused1QChain> fusion =
        findFusion(chain.operations, costModel, force);
    if (!fusion)
      continue;
    Operation *first = chain.operations.front();
    OpBuilder builder(first);
    Value output =
        emitFusion(builder, first->getLoc(), chain.inputs.front(), *fusion);
    cudaq::opt::replaceSingleQubitChain(std::move(chain), {output});
  }
}

namespace {

class Fuse1QToU3Pass : public cudaq::opt::impl::Fuse1QToU3Base<Fuse1QToU3Pass> {
public:
  using Fuse1QToU3Base::Fuse1QToU3Base;

  LogicalResult initialize(MLIRContext *context) override {
    if (!std::isfinite(epsilon) || epsilon < 0.0)
      return emitError(UnknownLoc::get(context),
                       "fuse-1q-to-u3 epsilon must be finite and non-negative");
    basisInfo.assign(basis.begin(), basis.end());
    if (cost == "gate-count") {
      const cudaq::detail::OperatorInfo u3("u3");
      const bool containsU3 =
          basisInfo.empty() ||
          llvm::any_of(basisInfo,
                       [&](const cudaq::detail::OperatorInfo &entry) {
                         return entry.covers(u3);
                       });
      costModel = std::make_shared<GateCountModel>(containsU3);
    } else if (cost == "clifford-t-free") {
      costModel = std::make_shared<CliffordTFreeModel>(epsilon);
    } else {
      return emitError(UnknownLoc::get(context),
                       "fuse-1q-to-u3 cost must be 'gate-count' or "
                       "'clifford-t-free'");
    }
    return success();
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (cudaq::runtime::isQuantumOptimizationDisabled(function))
      return;
    cudaq::opt::forEachNestedBlock(function.getBody(), [&](Block &block) {
      fuseBlock(block, *costModel, basisInfo);
    });
  }

private:
  /// Shared as the type must be copyable.
  std::shared_ptr<const CostModel> costModel;
  /// The parsed `basis` option.
  llvm::SmallVector<cudaq::detail::OperatorInfo> basisInfo;
};

} // namespace
