/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

inline Value createConstant(Location loc, double value, Type type,
                            PatternRewriter &rewriter) {
  auto fltTy = cast<FloatType>(type);
  return cudaq::opt::factory::createFloatConstant(loc, rewriter, value, fltTy);
}

inline Value createConstant(Location loc, std::size_t value,
                            PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
}

inline Value createDivF(Location loc, Value numerator, double denominator,
                        PatternRewriter &rewriter) {
  auto denominatorValue =
      createConstant(loc, denominator, numerator.getType(), rewriter);
  return rewriter.create<arith::DivFOp>(loc, numerator, denominatorValue);
}

/// @brief Returns true if \p op contains any `ControlType` operands.
inline bool containsControlTypes(quake::OperatorInterface op) {
  return llvm::any_of(op.getControls(), [](const Value &v) {
    return v.getType().isa<quake::ControlType>();
  });
}

/// @brief This is a wrapper class for `PatternRewriter::create<>()` for
/// `QuakeOperator`s. If the controls and targets are `quake::WireType`, then
/// this wrapper class's methods update the controls and targets in the `create`
/// calls to the corresponding wires in the output. If they are NOT `WireType`,
/// then the creates behave the exact same as a regular `PatternRewriter`.
class QuakeOperatorCreator {
public:
  QuakeOperatorCreator(PatternRewriter &rewriter) : rewriter(rewriter) {}

  /// Construct a resultType (suitable to be pass into the `TypeRange wires`
  /// builder for cases when you have one input ValueRange.
  SmallVector<Type> getResultType(ValueRange operands) {
    std::size_t numOutputWires = llvm::count_if(operands, [](const Value &v) {
      return v.getType().isa<quake::WireType>();
    });

    return SmallVector<Type>(numOutputWires,
                             quake::WireType::get(rewriter.getContext()));
  }

  /// Construct a resultType (suitable to be pass into the `TypeRange wires`
  /// builder for cases when you have two input ValueRanges.
  SmallVector<Type> getResultType(ValueRange operands1, ValueRange operands2) {
    std::size_t numOutputWires =
        llvm::count_if(
            operands1,
            [](const Value &v) { return v.getType().isa<quake::WireType>(); }) +
        llvm::count_if(operands2, [](const Value &v) {
          return v.getType().isa<quake::WireType>();
        });

    return SmallVector<Type>(numOutputWires,
                             quake::WireType::get(rewriter.getContext()));
  }

  /// Pluck out the values from \p newValues whose type is `WireType` and
  /// replace all the \p op uses with those values.
  void selectWiresAndReplaceUses(Operation *op, ValueRange newValues) {
    SmallVector<Value, 4> newWireValues;
    for (const auto &v : newValues)
      if (v.getType().isa<quake::WireType>())
        newWireValues.push_back(v);
    assert(op->getResults().size() == newWireValues.size() &&
           "incorrect number of output wires provided");
    op->replaceAllUsesWith(newWireValues);
  }

  /// Pluck out the values from \p controls and \p target whose type is
  /// `WireType` and replace all the \p op uses with those values.
  void selectWiresAndReplaceUses(Operation *op, ValueRange controls,
                                 Value target) {
    SmallVector<Value, 4> newWireValues;
    for (const auto &v : controls)
      if (v.getType().isa<quake::WireType>())
        newWireValues.push_back(v);
    if (target.getType().isa<quake::WireType>())
      newWireValues.push_back(target);
    assert(op->getResults().size() == newWireValues.size() &&
           "incorrect number of output wires provided");
    op->replaceAllUsesWith(newWireValues);
  }

  template <typename OpTy>
  OpTy create(Location location, Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(target), false,
                               ValueRange{}, ValueRange{}, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, bool is_adj, Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(target), is_adj,
                               ValueRange{}, ValueRange{}, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, Value &control, Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(control, target), false,
                               ValueRange{}, control, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (control.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      control = *resultIt++;
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, bool is_adj, ValueRange parameters,
              SmallVectorImpl<Value> &controls, Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(controls, target),
                               is_adj, parameters, controls, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (c.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
        c = *resultIt++;
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, ValueRange parameters,
              SmallVectorImpl<Value> &controls, Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(controls, target), false,
                               parameters, controls, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (c.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
        c = *resultIt++;
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, SmallVectorImpl<Value> &controls,
              Value &target) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(controls, target), false,
                               ValueRange{}, controls, target,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (c.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
        c = *resultIt++;
    if (target.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, SmallVectorImpl<Value> &targets) {
    OpTy op;
    op = rewriter.create<OpTy>(location, getResultType(targets), false,
                               ValueRange{}, ValueRange{}, targets,
                               DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &t : targets)
      if (t.getType().isa<quake::WireType>() && resultIt != resultWiresEnd)
        t = *resultIt++;
    return op;
  }

private:
  PatternRewriter &rewriter;
};

/// Check whether the operation has the correct number of controls.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
LogicalResult checkNumControls(quake::OperatorInterface op,
                               std::size_t requiredNumControls) {
  auto opControls = op.getControls();
  if (opControls.size() > requiredNumControls)
    return failure();

  // Compute the number of controls
  std::size_t numControls = 0;
  for (auto control : opControls) {
    if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      numControls += veq.getSize();
      continue;
    }
    numControls += 1;
  }

  return numControls == requiredNumControls ? success() : failure();
}

/// Check whether the operation has the correct number of controls. This
/// function take as input a mutable array reference, `controls`, which must
/// have the size equal to the number of controls. If the operation has `veq`s
/// as controls, split those into single qubit references.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
LogicalResult checkAndExtractControls(quake::OperatorInterface op,
                                      MutableArrayRef<Value> controls,
                                      PatternRewriter &rewriter) {
  if (failed(checkNumControls(op, controls.size())))
    return failure();

  std::size_t controlIndex = 0;
  for (Value control : op.getControls()) {
    if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
      for (std::size_t i = 0, end = veq.getSize(); i < end; ++i) {
        Value index = createConstant(op.getLoc(), i, rewriter);
        Value qref =
            rewriter.create<quake::ExtractRefOp>(op.getLoc(), control, index);
        controls[controlIndex] = qref;
        controlIndex += 1;
      }
    } else {
      controls[controlIndex] = control;
      controlIndex += 1;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// HOp decompositions
//===----------------------------------------------------------------------===//

// quake.h target
// ───────────────────────────────────
// quake.phased_rx(π/2, π/2) target
// quake.phased_rx(π, 0) target
struct HToPhasedRx : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern<quake::HOp>::OpRewritePattern;

  void initialize() { setDebugName("HToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::HOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);

    std::array<Value, 2> parameters = {pi_2, pi_2};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = pi;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.exp_pauli(theta) target pauliWord
// ───────────────────────────────────
// Basis change operations, cnots, rz(theta), adjoint basis change
struct ExpPauliDecomposition : public OpRewritePattern<quake::ExpPauliOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("ExpPauliDecomposition"); }

  LogicalResult matchAndRewrite(quake::ExpPauliOp expPauliOp,
                                PatternRewriter &rewriter) const override {
    auto loc = expPauliOp.getLoc();
    auto module = expPauliOp->getParentOfType<ModuleOp>();
    auto qubits = expPauliOp.getTarget();
    auto theta = expPauliOp.getParameter();
    auto pauliWord = expPauliOp.getPauli();

    if (expPauliOp.isAdj())
      theta = rewriter.create<arith::NegFOp>(loc, theta);

    std::optional<StringRef> optPauliWordStr;
    if (auto defOp =
            pauliWord.getDefiningOp<cudaq::cc::CreateStringLiteralOp>()) {
      optPauliWordStr = defOp.getStringLiteral();
    } else {
      // Get the pauli word string from a constant global string generated
      // during argument synthesis.
      auto stringOp = expPauliOp.getOperand(2);
      auto stringTy = stringOp.getType();
      if (auto charSpanTy = dyn_cast<cudaq::cc::CharspanType>(stringTy)) {
        if (auto vecInit = stringOp.getDefiningOp<cudaq::cc::StdvecInitOp>()) {
          auto addrOp = vecInit.getOperand(0);
          if (auto cast = addrOp.getDefiningOp<cudaq::cc::CastOp>())
            addrOp = cast.getOperand();
          if (auto addr = addrOp.getDefiningOp<cudaq::cc::AddressOfOp>()) {
            auto globalName = addr.getGlobalName();
            auto symbol = module.lookupSymbol(globalName);
            if (auto global = dyn_cast<LLVM::GlobalOp>(symbol)) {
              auto attr = global.getValue();
              auto strAttr = cast<mlir::StringAttr>(attr.value());
              optPauliWordStr = strAttr.getValue();
            }
          } else if (auto lit = addrOp.getDefiningOp<
                                cudaq::cc::CreateStringLiteralOp>()) {
            optPauliWordStr = lit.getStringLiteral();
          }
        }
      }
    }

    // Assert that we have a constant known pauli word
    if (!optPauliWordStr.has_value())
      return expPauliOp.emitOpError("cannot determine pauli word string");

    auto pauliWordStr = optPauliWordStr.value();

    // Remove optional last zero character
    auto size = pauliWordStr.size();
    if (size > 0 && pauliWordStr[size - 1] == '\0')
      size--;

    SmallVector<Value> qubitSupport;
    for (std::size_t i = 0; i < size; i++) {
      Value index = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
      Value qubitI = rewriter.create<quake::ExtractRefOp>(loc, qubits, index);
      if (pauliWordStr[i] != 'I')
        qubitSupport.push_back(qubitI);

      if (pauliWordStr[i] == 'Y') {
        APFloat d(M_PI_2);
        Value param = rewriter.create<arith::ConstantFloatOp>(
            loc, d, rewriter.getF64Type());
        rewriter.create<quake::RxOp>(loc, ValueRange{param}, ValueRange{},
                                     ValueRange{qubitI});
      } else if (pauliWordStr[i] == 'X') {
        rewriter.create<quake::HOp>(loc, ValueRange{qubitI});
      }
    }

    // If qubitSupport is empty, then we can safely drop the
    // operation since it will only add a global phase.
    // FIXME this should be tracked in the IR at some point
    if (qubitSupport.empty()) {
      rewriter.eraseOp(expPauliOp);
      return success();
    }

    std::vector<std::pair<Value, Value>> toReverse;
    for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
      rewriter.create<quake::XOp>(loc, ValueRange{qubitSupport[i]},
                                  ValueRange{qubitSupport[i + 1]});
      toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
    }

    rewriter.create<quake::RzOp>(loc, ValueRange{theta}, ValueRange{},
                                 ValueRange{qubitSupport.back()});

    std::reverse(toReverse.begin(), toReverse.end());
    for (auto &[i, j] : toReverse)
      rewriter.create<quake::XOp>(loc, ValueRange{i}, ValueRange{j});

    for (std::size_t i = 0; i < pauliWordStr.size(); i++) {
      std::size_t k = pauliWordStr.size() - 1 - i;
      Value index = rewriter.create<arith::ConstantIntOp>(loc, k, 64);
      Value qubitK = rewriter.create<quake::ExtractRefOp>(loc, qubits, index);

      if (pauliWordStr[k] == 'Y') {
        APFloat d(-M_PI_2);
        Value param = rewriter.create<arith::ConstantFloatOp>(
            loc, d, rewriter.getF64Type());
        rewriter.create<quake::RxOp>(loc, ValueRange{param}, ValueRange{},
                                     ValueRange{qubitK});
      } else if (pauliWordStr[k] == 'X') {
        rewriter.create<quake::HOp>(loc, ValueRange{qubitK});
      }
    }

    rewriter.eraseOp(expPauliOp);

    return success();
  }
};

// Naive mapping of R1 to Rz, ignoring the global phase.
// This is only expected to work with full inlining and
// quake apply specialization.
struct R1ToRz : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    if (!r1Op.getControls().empty())
      return failure();

    rewriter.replaceOpWithNewOp<quake::RzOp>(
        r1Op, r1Op.isAdj(), r1Op.getParameters(), r1Op.getControls(),
        r1Op.getTargets());
    return success();
  }
};

// Naive mapping of R1 to U3
// quake.r1(λ) [control] target
// ───────────────────────────────────
// quake.u3(0, 0, λ) [control] target
struct R1ToU3 : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("R1ToU3"); }

  LogicalResult matchAndRewrite(quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    Location loc = r1Op->getLoc();
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    std::array<Value, 3> parameters = {zero, zero, r1Op.getParameters()[0]};
    rewriter.replaceOpWithNewOp<quake::U3Op>(
        r1Op, r1Op.isAdj(), parameters, r1Op.getControls(), r1Op.getTargets());
    return success();
  }
};

// quake.r1<adj> (θ) target
// ─────────────────────────────────
// quake.r1(-θ) target
struct R1AdjToR1 : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("R1AdjToR1"); }

  LogicalResult matchAndRewrite(quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::R1Op>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.swap a, b
// ───────────────────────────────────
// quake.cnot b, a;
// quake.cnot a, b;
// quake.cnot b, a;
struct SwapToCX : public OpRewritePattern<quake::SwapOp> {
  using OpRewritePattern<quake::SwapOp>::OpRewritePattern;

  void initialize() { setDebugName("SwapToCX"); }

  LogicalResult matchAndRewrite(quake::SwapOp op,
                                PatternRewriter &rewriter) const override {
    // Op info
    Location loc = op->getLoc();
    Value a = op.getTarget(0);
    Value b = op.getTarget(1);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::XOp>(loc, b, a);
    qRewriter.create<quake::XOp>(loc, a, b);
    qRewriter.create<quake::XOp>(loc, b, a);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{a, b});
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.h control, target
// ───────────────────────────────────
// quake.s target;
// quake.h target;
// quake.t target;
// quake.x control, target;
// quake.t<adj> target;
// quake.h target;
// quake.s<adj> target;
struct CHToCX : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern<quake::HOp>::OpRewritePattern;

  void initialize() { setDebugName("CHToCX"); }

  LogicalResult matchAndRewrite(quake::HOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value control = op.getControls()[0];
    Value target = op.getTarget();

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::SOp>(loc, target);
    qRewriter.create<quake::HOp>(loc, target);
    qRewriter.create<quake::TOp>(loc, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/true, target);
    qRewriter.create<quake::HOp>(loc, target);
    qRewriter.create<quake::SOp>(loc, /*isAdj=*/true, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SOp decompositions
//===----------------------------------------------------------------------===//

// quake.s target
// ──────────────────────────────
// phased_rx(π/2, 0) target
// phased_rx(-π/2, π/2) target
// phased_rx(-π/2, 0) target
struct SToPhasedRx : public OpRewritePattern<quake::SOp> {
  using OpRewritePattern<quake::SOp>::OpRewritePattern;

  void initialize() { setDebugName("SToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::SOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    Value angle = op.isAdj() ? pi_2 : negPi_2;

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.s [control] target
// ────────────────────────────────────
// quake.r1(π/2) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct SToR1 : public OpRewritePattern<quake::SOp> {
  using OpRewritePattern<quake::SOp>::OpRewritePattern;

  void initialize() { setDebugName("SToR1"); }

  LogicalResult matchAndRewrite(quake::SOp op,
                                PatternRewriter &rewriter) const override {
    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_2 : M_PI_2,
                                rewriter.getF64Type(), rewriter);

    SmallVector<Value> controls(op.getControls());
    Value target = op.getTarget();
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::R1Op>(loc, angle, controls, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOp decompositions
//===----------------------------------------------------------------------===//

// quake.t target
// ────────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π/4, π/2) target
// quake.phased_rx(-π/2, 0) target
struct TToPhasedRx : public OpRewritePattern<quake::TOp> {
  using OpRewritePattern<quake::TOp>::OpRewritePattern;

  void initialize() { setDebugName("TToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::TOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = createConstant(loc, -M_PI_4, rewriter.getF64Type(), rewriter);
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.t [control] target
// ────────────────────────────────────
// quake.r1(π/4) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct TToR1 : public OpRewritePattern<quake::TOp> {
  using OpRewritePattern<quake::TOp>::OpRewritePattern;

  void initialize() { setDebugName("TToR1"); }

  LogicalResult matchAndRewrite(quake::TOp op,
                                PatternRewriter &rewriter) const override {
    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_4 : M_PI_4,
                                rewriter.getF64Type(), rewriter);
    SmallVector<Value> controls(op.getControls());
    Value target = op.getTarget();
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::R1Op>(loc, angle, controls, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XOp decompositions
//===----------------------------------------------------------------------===//

// quake.x [control] target
// ──────────────────────────────────
// quake.h target
// quake.z [control] target
// quake.h target
struct CXToCZ : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("CXToCZ"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkNumControls(op, 1)))
      return failure();
    // This decomposition does not support `quake.control` types because the
    // input controls are used as targets during this transformation.
    if (containsControlTypes(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    SmallVector<Value> controls = op.getControls();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::HOp>(loc, target);
    if (negControl)
      qRewriter.create<quake::XOp>(loc, controls);
    qRewriter.create<quake::ZOp>(loc, controls, target);
    if (negControl)
      qRewriter.create<quake::XOp>(loc, controls);
    qRewriter.create<quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.x [controls] target
// ──────────────────────────────────
// quake.h target
// quake.z [controls] target
// quake.h target
struct CCXToCCZ : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("CCXToCCZ"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkNumControls(op, 2)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    SmallVector<Value> controls = op.getControls();

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::HOp>(loc, target);
    auto zOp = qRewriter.create<quake::ZOp>(loc, controls, target);
    zOp.setNegatedQubitControls(op.getNegatedQubitControls());
    qRewriter.create<quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.x target
// ───────────────────────────────
// quake.phased_rx(π, 0) target
struct XToPhasedRx : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("XToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);

    SmallVector<Value> parameters = {pi, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// YOp decompositions
//===----------------------------------------------------------------------===//

// quake.y target
// ─────────────────────────────────
// quake.phased_rx(π, -π/2) target
struct YToPhasedRx : public OpRewritePattern<quake::YOp> {
  using OpRewritePattern<quake::YOp>::OpRewritePattern;

  void initialize() { setDebugName("YToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::YOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);
    Value negPi_2 =
        createConstant(loc, -M_PI_2, rewriter.getF64Type(), rewriter);

    SmallVector<Value> parameters = {pi, negPi_2};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ZOp decompositions
//===----------------------------------------------------------------------===//

//                                                                  ┌───┐
//  ───●────  ──────────────●───────────────────●──────●─────────●──┤ T ├
//     │                    │                   │      │         │  └───┘
//     │                    │                   │    ┌─┴─┐┌───┐┌─┴─┐┌───┐
//  ───●─── = ────●─────────┼─────────●─────────┼────┤ X ├┤ ┴ ├┤ X ├┤ T ├
//     │          │         │         │         │    └───┘└───┘└───┘└───┘
//   ┌─┴─┐      ┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐                 ┌───┐
//  ─┤ z ├─   ──┤ X ├┤ ┴ ├┤ X ├┤ T ├┤ X ├┤ ┴ ├┤ X ├─────────────────┤ T ├
//   └───┘      └───┘└───┘└───┘└───┘└───┘└───┘└───┘                 └───┘
//
// NOTE: `┴` denotes the adjoint of `T`.
struct CCZToCX : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("CCZToCX"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    // This decomposition does not support `quake.control` types because the
    // input controls are used as targets during this transformation.
    if (containsControlTypes(op))
      return failure();

    SmallVector<Value, 2> controls(2);
    if (failed(checkAndExtractControls(op, controls, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negC0 = false;
    auto negC1 = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls) {
      negC0 = (*negatedControls)[0];
      negC1 = (*negatedControls)[1];
      // The order of controls don't matter for the operation. However, this
      // pattern relies on a normalization: if only one control is complemented,
      // it must be the 0th one, which means that a negated 1th control implies
      // a negated 0th. This normalization allow us to decompose more
      // straightforwardly.
      if (!negC0 && negC1) {
        negC0 = true;
        negC1 = false;
        std::swap(controls[0], controls[1]);
      }
    }

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::XOp>(loc, controls[1], target);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/!negC0, target);
    qRewriter.create<quake::XOp>(loc, controls[0], target);
    qRewriter.create<quake::TOp>(loc, target);
    qRewriter.create<quake::XOp>(loc, controls[1], target);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/!negC1, target);
    qRewriter.create<quake::XOp>(loc, controls[0], target);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/negC0 && !negC1, target);

    qRewriter.create<quake::XOp>(loc, controls[0], controls[1]);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/true, controls[1]);
    qRewriter.create<quake::XOp>(loc, controls[0], controls[1]);
    qRewriter.create<quake::TOp>(loc, /*isAdj=*/negC0, controls[1]);

    qRewriter.create<quake::TOp>(loc, /*isAdj=*/negC1, controls[0]);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.z [control] target
// ──────────────────────────────────
// quake.h target
// quake.x [control] target
// quake.h target
struct CZToCX : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("CZToCX"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    // This decomposition does not support `quake.control` types because the
    // input controls are used as targets during this transformation.
    if (containsControlTypes(op))
      return failure();
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    SmallVector<Value> controls(op.getControls());
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::HOp>(loc, target);
    if (negControl)
      qRewriter.create<quake::XOp>(loc, controls);
    qRewriter.create<quake::XOp>(loc, controls, target);
    if (negControl)
      qRewriter.create<quake::XOp>(loc, controls);
    qRewriter.create<quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.z target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π, π/2) target
// quake.phased_rx(-π/2, 0) target
struct ZToPhasedRx : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("ZToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value negPi = createConstant(loc, -M_PI, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi;
    parameters[1] = pi_2;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// R1Op decompositions
//===----------------------------------------------------------------------===//

// quake.r1(λ) [control] target
// ───────────────────────────────
// quake.r1(λ/2) control
// quake.x [control] target
// quake.r1(-λ/2) target
// quake.x [control] target
// quake.r1(λ/2) target
struct CR1ToCX : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("CR1ToCX"); }

  LogicalResult matchAndRewrite(quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (containsControlTypes(op))
      return failure();

    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::R1Op>(loc, /*isAdj*/ negControl, halfAngle,
                                  noControls, control);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::R1Op>(loc, /*isAdj*/ negControl, negHalfAngle,
                                  noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::R1Op>(loc, halfAngle, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.r1(λ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-λ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct R1ToPhasedRx : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("R1ToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);
    Value negAngle = rewriter.create<arith::NegFOp>(loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RxOp decompositions
//===----------------------------------------------------------------------===//

// quake.rx(θ) [control] target
// ───────────────────────────────
// quake.s target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
// quake.ry(θ/2) target
// quake.rz(-π/2) target
struct CRxToCX : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern<quake::RxOp>::OpRewritePattern;

  void initialize() { setDebugName("CRxToCX"); }

  LogicalResult matchAndRewrite(quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();
    assert(control);

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);
    Value negPI_2 = createConstant(loc, -M_PI_2, angleType, rewriter);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::SOp>(loc, /*isAdj*/ negControl, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::RyOp>(loc, negHalfAngle, noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::RyOp>(loc, /*isAdj*/ negControl, halfAngle,
                                  noControls, target);
    qRewriter.create<quake::RzOp>(loc, /*isAdj*/ negControl, negPI_2,
                                  noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rx(θ) target
// ───────────────────────────────
// quake.phased_rx(θ, 0) target
struct RxToPhasedRx : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern<quake::RxOp>::OpRewritePattern;

  void initialize() { setDebugName("RxToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);

    SmallVector<Value> parameters = {angle, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rx<adj> (θ) target
// ─────────────────────────────────
// quake.rx(-θ) target
struct RxAdjToRx : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern<quake::RxOp>::OpRewritePattern;

  void initialize() { setDebugName("RxAdjToRx"); }

  LogicalResult matchAndRewrite(quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RyOp decompositions
//===----------------------------------------------------------------------===//

// quake.ry(θ) [control] target
// ───────────────────────────────
// quake.ry(θ/2) target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
struct CRyToCX : public OpRewritePattern<quake::RyOp> {
  using OpRewritePattern<quake::RyOp>::OpRewritePattern;

  void initialize() { setDebugName("CRyToCX"); }

  LogicalResult matchAndRewrite(quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RyOp>(loc, halfAngle, noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::RyOp>(loc, /*isAdj*/ negControl, negHalfAngle,
                                  noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.ry(θ) target
// ─────────────────────────────────
// quake.phased_rx(θ, π/2) target
struct RyToPhasedRx : public OpRewritePattern<quake::RyOp> {
  using OpRewritePattern<quake::RyOp>::OpRewritePattern;

  void initialize() { setDebugName("RyToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);

    SmallVector<Value> parameters = {angle, pi_2};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.ry<adj> (θ) target
// ─────────────────────────────────
// quake.ry(-θ) target
struct RyAdjToRy : public OpRewritePattern<quake::RyOp> {
  using OpRewritePattern<quake::RyOp>::OpRewritePattern;

  void initialize() { setDebugName("RyAdjToRy"); }

  LogicalResult matchAndRewrite(quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RyOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RzOp decompositions
//===----------------------------------------------------------------------===//

// quake.rz(λ) [control] target
// ───────────────────────────────
// quake.rz(λ/2) target
// quake.x [control] target
// quake.rz(-λ/2) target
// quake.x [control] target
struct CRzToCX : public OpRewritePattern<quake::RzOp> {
  using OpRewritePattern<quake::RzOp>::OpRewritePattern;

  void initialize() { setDebugName("CRzToCX"); }

  LogicalResult matchAndRewrite(quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RzOp>(loc, halfAngle, noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);
    qRewriter.create<quake::RzOp>(loc, /*isAdj*/ negControl, negHalfAngle,
                                  noControls, target);
    qRewriter.create<quake::XOp>(loc, control, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rz(θ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-θ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct RzToPhasedRx : public OpRewritePattern<quake::RzOp> {
  using OpRewritePattern<quake::RzOp>::OpRewritePattern;

  void initialize() { setDebugName("RzToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);
    Value negAngle = rewriter.create<arith::NegFOp>(loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rz<adj> (θ) target
// ─────────────────────────────────
// quake.rz(-θ) target
struct RzAdjToRz : public OpRewritePattern<quake::RzOp> {
  using OpRewritePattern<quake::RzOp>::OpRewritePattern;

  void initialize() { setDebugName("RzAdjToRz"); }

  LogicalResult matchAndRewrite(quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RzOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// U3Op decompositions
//===----------------------------------------------------------------------===//

// quake.u3(θ,ϕ,λ) target
// ──────────────────────────────────
// quake.rz(λ) target
// quake.rx(π/2) target
// quake.rz(θ) target
// quake.rx(-π/2) target
// quake.rz(ϕ) target
struct U3ToRotations : public OpRewritePattern<quake::U3Op> {
  using OpRewritePattern<quake::U3Op>::OpRewritePattern;

  void initialize() { setDebugName("U3ToRotations"); }

  LogicalResult matchAndRewrite(quake::U3Op op,
                                PatternRewriter &rewriter) const override {
    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    SmallVector<Value> controls(op.getControls());
    Value theta = op.getParameters()[0];
    Value phi = op.getParameters()[1];
    Value lam = op.getParameters()[2];

    if (op.isAdj()) {
      theta = rewriter.create<arith::NegFOp>(loc, theta);
      phi = rewriter.create<arith::NegFOp>(loc, phi);
      lam = rewriter.create<arith::NegFOp>(loc, lam);
    }

    // Necessary/Helpful constants
    Type angleType = op.getParameter().getType();
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::RzOp>(loc, lam, controls, target);
    qRewriter.create<quake::RxOp>(loc, pi_2, controls, target);
    qRewriter.create<quake::RzOp>(loc, theta, controls, target);
    qRewriter.create<quake::RxOp>(loc, negPi_2, controls, target);
    qRewriter.create<quake::RzOp>(loc, phi, controls, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Populating pattern sets
//===----------------------------------------------------------------------===//

void cudaq::populateWithAllDecompositionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.insert<
    // HOp patterns
    HToPhasedRx,
    CHToCX,
    // SOp patterns
    SToPhasedRx,
    SToR1,
    // TOp patterns
    TToPhasedRx,
    TToR1,
    // XOp patterns
    CXToCZ,
    CCXToCCZ,
    XToPhasedRx,
    // YOp patterns
    YToPhasedRx,
    // ZOp patterns
    CZToCX,
    CCZToCX,
    ZToPhasedRx,
    // R1Op patterns
    CR1ToCX,
    R1ToPhasedRx,
    R1ToRz,
    R1ToU3,
    R1AdjToR1,
    // RxOp patterns
    CRxToCX,
    RxToPhasedRx,
    RxAdjToRx,
    // RyOp patterns
    CRyToCX,
    RyToPhasedRx,
    RyAdjToRy,
    // RzOp patterns
    CRzToCX,
    RzToPhasedRx,
    RzAdjToRz,
    // Swap
    SwapToCX,
    // U3Op
    U3ToRotations,
    ExpPauliDecomposition
  >(patterns.getContext());
  // clang-format on
}
