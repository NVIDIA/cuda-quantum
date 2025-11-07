/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/**
 * This file contains the decomposition patterns that match single gates and
 * decompose them into a sequence of other gates.
 *
 * Each pattern definition contains 3 elements:
 * 1. The pattern type, which contains the pattern metadata. It must inherit
 * from DecompositionPatternType.
 * 2. The pattern itself, which defines what ops to match and how to replace
 * them. It must inherit from DecompositionPattern<PatternType, Op>.
 * 3. A call to the CUDAQ_REGISTER_TYPE macro to register the pattern in the
 * registry.
 */

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TypeName.h>
#include <memory>

using namespace mlir;

LLVM_INSTANTIATE_REGISTRY(cudaq::DecompositionPatternType::RegistryType)

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

// From here on, we define the decomposition patterns ==========================

// TODO: "SToR1", "TToR1", "R1ToU3", "U3ToRotations" can be generalised
// arbitrary number of controls, but we would need to reason over n HOp patterns

//===----------------------------------------------------------------------===//
// HOp decompositions
//===----------------------------------------------------------------------===//

// quake.h target
// ───────────────────────────────────
// quake.phased_rx(π/2, π/2) target
// quake.phased_rx(π, 0) target

struct HToPhasedRxType;
struct HToPhasedRx
    : public cudaq::DecompositionPattern<HToPhasedRxType, quake::HOp> {

  using cudaq::DecompositionPattern<HToPhasedRxType,
                                    quake::HOp>::DecompositionPattern;

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
struct HToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;

  llvm::StringRef getSourceOp() const override { return "h"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "HToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<HToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, HToPhasedRxType,
                    HToPhasedRx);

// quake.exp_pauli(theta) target pauliWord
// ───────────────────────────────────
// Basis change operations, cnots, rz(theta), adjoint basis change
struct ExpPauliDecompositionType;
struct ExpPauliDecomposition
    : public cudaq::DecompositionPattern<ExpPauliDecompositionType,
                                         quake::ExpPauliOp> {
  using cudaq::DecompositionPattern<ExpPauliDecompositionType,
                                    quake::ExpPauliOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(quake::ExpPauliOp expPauliOp,
                                PatternRewriter &rewriter) const override {
    auto loc = expPauliOp.getLoc();
    auto module = expPauliOp->getParentOfType<ModuleOp>();
    auto qubits = expPauliOp.getTarget();
    auto theta = expPauliOp.getParameter();
    auto pauliWord = expPauliOp.getPauli();

    if (expPauliOp.isAdj())
      theta = rewriter.create<arith::NegFOp>(loc, theta);

    std::optional<std::string> optPauliWordStr;
    if (!pauliWord) {
      optPauliWordStr = expPauliOp.getPauliLiteral()->str();
    } else {
      Type stringTy = pauliWord.getType();
      if (isa<cudaq::cc::PointerType>(stringTy)) {
        if (auto defOp =
                pauliWord.getDefiningOp<cudaq::cc::CreateStringLiteralOp>())
          optPauliWordStr = defOp.getStringLiteral();
      } else {
        if (auto charSpanTy = dyn_cast<cudaq::cc::CharspanType>(stringTy)) {
          if (auto load = pauliWord.getDefiningOp<cudaq::cc::LoadOp>()) {
            // Look for a matching StoreOp for the LoadOp. This search isn't
            // necessarily efficient or exhaustive. Instead of using dominance
            // information, we scan the current basic block looking for the
            // nearest StoreOp before the LoadOp. If one is found, we forward
            // the stored value.
            auto ptrVal = load.getPtrvalue();
            auto storeVal = [&]() -> Value {
              SmallVector<Operation *> stores;
              for (auto *use : ptrVal.getUsers()) {
                if (auto store = dyn_cast<cudaq::cc::StoreOp>(use)) {
                  if (store.getPtrvalue() == ptrVal &&
                      store->getBlock() == load->getBlock())
                    stores.push_back(store.getOperation());
                }
              }
              if (stores.empty())
                return {};
              for (Operation *op = load.getOperation()->getPrevNode(); op;
                   op = op->getPrevNode()) {
                auto iter = std::find(stores.begin(), stores.end(), op);
                if (iter == stores.end())
                  continue;
                return cast<cudaq::cc::StoreOp>(*iter).getValue();
              }
              return {};
            }();
            if (storeVal)
              pauliWord = storeVal;
          }
          if (auto vecInit =
                  pauliWord.getDefiningOp<cudaq::cc::StdvecInitOp>()) {
            auto addrOp = vecInit.getOperand(0);
            if (auto cast = addrOp.getDefiningOp<cudaq::cc::CastOp>())
              addrOp = cast.getOperand();
            if (auto addr = addrOp.getDefiningOp<cudaq::cc::AddressOfOp>()) {
              // Get the pauli word string from a constant global string
              // generated during argument synthesis.
              auto globalName = addr.getGlobalName();
              auto symbol = module.lookupSymbol(globalName);
              if (auto global = dyn_cast<LLVM::GlobalOp>(symbol)) {
                auto attr = global.getValue();
                auto strAttr = cast<mlir::StringAttr>(attr.value());
                optPauliWordStr = strAttr.getValue();
              } else if (auto global = dyn_cast<cudaq::cc::GlobalOp>(symbol)) {
                auto attr = global.getValue();
                auto elementsAttr = cast<mlir::ElementsAttr>(attr.value());
                auto eleTy = elementsAttr.getElementType();
                auto values = elementsAttr.getValues<mlir::Attribute>();

                std::string pauliWordString;
                pauliWordString.reserve(values.size());
                for (auto it = values.begin(); it != values.end(); ++it) {
                  assert(isa<IntegerType>(eleTy));
                  char v = static_cast<char>(cast<IntegerAttr>(*it).getInt());
                  pauliWordString.push_back(v);
                }
                optPauliWordStr = StringRef(pauliWordString);
              }
            } else if (auto lit = addrOp.getDefiningOp<
                                  cudaq::cc::CreateStringLiteralOp>()) {
              // Get the pauli word string if it was a literal wrapped in a
              // stdvec structure.
              optPauliWordStr = lit.getStringLiteral();
            }
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
struct ExpPauliDecompositionType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "exp_pauli"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rx", "h", "x(1)", "rz"};
    return ops;
  }
  llvm::StringRef getPatternName() const override {
    return "ExpPauliDecomposition";
  }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<ExpPauliDecomposition>(context, benefit);
    return pattern;
  }
};

CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, ExpPauliDecompositionType,
                    ExpPauliDecomposition);

// Naive mapping of R1 to Rz, ignoring the global phase.
// This is only expected to work with full inlining and
// quake apply specialization.
struct R1ToRzType;
struct R1ToRz : public cudaq::DecompositionPattern<R1ToRzType, quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToRzType,
                                    quake::R1Op>::DecompositionPattern;

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
struct R1ToRzType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "r1"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rz"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "R1ToRz"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<R1ToRz>(context, benefit);
    return pattern;
  }
};

CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, R1ToRzType, R1ToRz);

// Naive mapping of R1 to U3
// quake.r1(λ) [control] target
// ───────────────────────────────────
// quake.u3(0, 0, λ) [control] target
struct R1ToU3Type;
struct R1ToU3 : public cudaq::DecompositionPattern<R1ToU3Type, quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToU3Type,
                                    quake::R1Op>::DecompositionPattern;

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
struct R1ToU3Type : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "r1"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"u3"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "R1ToU3"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<R1ToU3>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, R1ToU3Type, R1ToU3);

// quake.r1<adj> (θ) target
// ─────────────────────────────────
// quake.r1(-θ) target
struct R1AdjToR1Type;
struct R1AdjToR1
    : public cudaq::DecompositionPattern<R1AdjToR1Type, quake::R1Op> {
  using cudaq::DecompositionPattern<R1AdjToR1Type,
                                    quake::R1Op>::DecompositionPattern;

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
struct R1AdjToR1Type : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "r1"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"r1"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "R1AdjToR1"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<R1AdjToR1>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, R1AdjToR1Type, R1AdjToR1);

// quake.swap a, b
// ───────────────────────────────────
// quake.cnot b, a;
// quake.cnot a, b;
// quake.cnot b, a;
struct SwapToCXType;
struct SwapToCX
    : public cudaq::DecompositionPattern<SwapToCXType, quake::SwapOp> {
  using cudaq::DecompositionPattern<SwapToCXType,
                                    quake::SwapOp>::DecompositionPattern;

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
struct SwapToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "swap"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "SwapToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<SwapToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, SwapToCXType, SwapToCX);

// quake.h control, target
// ───────────────────────────────────
// quake.s target;
// quake.h target;
// quake.t target;
// quake.x control, target;
// quake.t<adj> target;
// quake.h target;
// quake.s<adj> target;
struct CHToCXType;
struct CHToCX : public cudaq::DecompositionPattern<CHToCXType, quake::HOp> {
  using cudaq::DecompositionPattern<CHToCXType,
                                    quake::HOp>::DecompositionPattern;

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
struct CHToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "h(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"s", "h", "t", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CHToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CHToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CHToCXType, CHToCX);

//===----------------------------------------------------------------------===//
// SOp decompositions
//===----------------------------------------------------------------------===//

// quake.s target
// ──────────────────────────────
// phased_rx(π/2, 0) target
// phased_rx(-π/2, π/2) target
// phased_rx(-π/2, 0) target
struct SToPhasedRxType;
struct SToPhasedRx
    : public cudaq::DecompositionPattern<SToPhasedRxType, quake::SOp> {
  using cudaq::DecompositionPattern<SToPhasedRxType,
                                    quake::SOp>::DecompositionPattern;

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
struct SToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "s"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "SToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<SToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, SToPhasedRxType,
                    SToPhasedRx);

// quake.s [control] target
// ────────────────────────────────────
// quake.r1(π/2) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct SToR1Type;
struct SToR1 : public cudaq::DecompositionPattern<SToR1Type, quake::SOp> {
  using cudaq::DecompositionPattern<SToR1Type,
                                    quake::SOp>::DecompositionPattern;

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
struct SToR1Type : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "s"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"r1"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "SToR1"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<SToR1>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, SToR1Type, SToR1);

//===----------------------------------------------------------------------===//
// TOp decompositions
//===----------------------------------------------------------------------===//

// quake.t target
// ────────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π/4, π/2) target
// quake.phased_rx(-π/2, 0) target
struct TToPhasedRxType;
struct TToPhasedRx
    : public cudaq::DecompositionPattern<TToPhasedRxType, quake::TOp> {
  using cudaq::DecompositionPattern<TToPhasedRxType,
                                    quake::TOp>::DecompositionPattern;

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
struct TToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "t"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "TToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<TToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, TToPhasedRxType,
                    TToPhasedRx);

// quake.t [control] target
// ────────────────────────────────────
// quake.r1(π/4) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct TToR1Type;
struct TToR1 : public cudaq::DecompositionPattern<TToR1Type, quake::TOp> {
  using cudaq::DecompositionPattern<TToR1Type,
                                    quake::TOp>::DecompositionPattern;

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
struct TToR1Type : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "t"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"r1"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "TToR1"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<TToR1>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, TToR1Type, TToR1);

//===----------------------------------------------------------------------===//
// XOp decompositions
//===----------------------------------------------------------------------===//

// quake.x [control] target
// ──────────────────────────────────
// quake.h target
// quake.z [control] target
// quake.h target
struct CXToCZType;
struct CXToCZ : public cudaq::DecompositionPattern<CXToCZType, quake::XOp> {
  using cudaq::DecompositionPattern<CXToCZType,
                                    quake::XOp>::DecompositionPattern;

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
struct CXToCZType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "x(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"h", "z(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CXToCZ"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CXToCZ>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CXToCZType, CXToCZ);

// quake.x [controls] target
// ──────────────────────────────────
// quake.h target
// quake.z [controls] target
// quake.h target
struct CCXToCCZType;
struct CCXToCCZ : public cudaq::DecompositionPattern<CCXToCCZType, quake::XOp> {
  using cudaq::DecompositionPattern<CCXToCCZType,
                                    quake::XOp>::DecompositionPattern;

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
struct CCXToCCZType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "x(2)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"h", "z(2)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CCXToCCZ"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CCXToCCZ>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CCXToCCZType, CCXToCCZ);

// quake.x target
// ───────────────────────────────
// quake.phased_rx(π, 0) target
struct XToPhasedRxType;
struct XToPhasedRx
    : public cudaq::DecompositionPattern<XToPhasedRxType, quake::XOp> {
  using cudaq::DecompositionPattern<XToPhasedRxType,
                                    quake::XOp>::DecompositionPattern;

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
struct XToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "x"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "XToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<XToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, XToPhasedRxType,
                    XToPhasedRx);

//===----------------------------------------------------------------------===//
// YOp decompositions
//===----------------------------------------------------------------------===//

// quake.y target
// ─────────────────────────────────
// quake.phased_rx(π, -π/2) target
struct YToPhasedRxType;
struct YToPhasedRx
    : public cudaq::DecompositionPattern<YToPhasedRxType, quake::YOp> {
  using cudaq::DecompositionPattern<YToPhasedRxType,
                                    quake::YOp>::DecompositionPattern;

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
struct YToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "y"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "YToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<YToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, YToPhasedRxType,
                    YToPhasedRx);

// quake.y [control] target
// ───────────────────────────────────
// quake.s<adj> target;
// quake.x [control] target;
// quake.s target;

struct CYToCXType;
struct CYToCX : public cudaq::DecompositionPattern<CYToCXType, quake::YOp> {
  using cudaq::DecompositionPattern<CYToCXType,
                                    quake::YOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(quake::YOp op,
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

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<quake::SOp>(loc, /*isAdj=*/true, target);
    qRewriter.create<quake::XOp>(loc, controls, target);
    qRewriter.create<quake::SOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
struct CYToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "y(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"s", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CYToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CYToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CYToCXType, CYToCX);

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
struct CCZToCXType;
struct CCZToCX : public cudaq::DecompositionPattern<CCZToCXType, quake::ZOp> {
  using cudaq::DecompositionPattern<CCZToCXType,
                                    quake::ZOp>::DecompositionPattern;

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
struct CCZToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "z(2)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"t", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CCZToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CCZToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CCZToCXType, CCZToCX);

// quake.z [control] target
// ──────────────────────────────────
// quake.h target
// quake.x [control] target
// quake.h target

struct CZToCXType;
struct CZToCX : public cudaq::DecompositionPattern<CZToCXType, quake::ZOp> {
  using cudaq::DecompositionPattern<CZToCXType,
                                    quake::ZOp>::DecompositionPattern;

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
struct CZToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "z(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"h", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CZToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CZToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CZToCXType, CZToCX);

// quake.z target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π, π/2) target
// quake.phased_rx(-π/2, 0) target
struct ZToPhasedRxType;
struct ZToPhasedRx
    : public cudaq::DecompositionPattern<ZToPhasedRxType, quake::ZOp> {
  using cudaq::DecompositionPattern<ZToPhasedRxType,
                                    quake::ZOp>::DecompositionPattern;

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
struct ZToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "z"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "ZToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<ZToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, ZToPhasedRxType,
                    ZToPhasedRx);

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
struct CR1ToCXType;
struct CR1ToCX : public cudaq::DecompositionPattern<CR1ToCXType, quake::R1Op> {
  using cudaq::DecompositionPattern<CR1ToCXType,
                                    quake::R1Op>::DecompositionPattern;

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
struct CR1ToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "r1(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"r1", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CR1ToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CR1ToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CR1ToCXType, CR1ToCX);

// quake.r1(λ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-λ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct R1ToPhasedRxType;
struct R1ToPhasedRx
    : public cudaq::DecompositionPattern<R1ToPhasedRxType, quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToPhasedRxType,
                                    quake::R1Op>::DecompositionPattern;

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
struct R1ToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "r1"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "R1ToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<R1ToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, R1ToPhasedRxType,
                    R1ToPhasedRx);

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
struct CRxToCXType;
struct CRxToCX : public cudaq::DecompositionPattern<CRxToCXType, quake::RxOp> {
  using cudaq::DecompositionPattern<CRxToCXType,
                                    quake::RxOp>::DecompositionPattern;

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
struct CRxToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rx(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"s", "x(1)", "ry", "rz"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CRxToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CRxToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CRxToCXType, CRxToCX);

// quake.rx(θ) target
// ───────────────────────────────
// quake.phased_rx(θ, 0) target
struct RxToPhasedRxType;
struct RxToPhasedRx
    : public cudaq::DecompositionPattern<RxToPhasedRxType, quake::RxOp> {
  using cudaq::DecompositionPattern<RxToPhasedRxType,
                                    quake::RxOp>::DecompositionPattern;

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
struct RxToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rx"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RxToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RxToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RxToPhasedRxType,
                    RxToPhasedRx);

// quake.rx<adj> (θ) target
// ─────────────────────────────────
// quake.rx(-θ) target
struct RxAdjToRxType;
struct RxAdjToRx
    : public cudaq::DecompositionPattern<RxAdjToRxType, quake::RxOp> {
  using cudaq::DecompositionPattern<RxAdjToRxType,
                                    quake::RxOp>::DecompositionPattern;

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
struct RxAdjToRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rx"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RxAdjToRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RxAdjToRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RxAdjToRxType, RxAdjToRx);

//===----------------------------------------------------------------------===//
// RyOp decompositions
//===----------------------------------------------------------------------===//

// quake.ry(θ) [control] target
// ───────────────────────────────
// quake.ry(θ/2) target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
struct CRyToCXType;
struct CRyToCX : public cudaq::DecompositionPattern<CRyToCXType, quake::RyOp> {
  using cudaq::DecompositionPattern<CRyToCXType,
                                    quake::RyOp>::DecompositionPattern;

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
struct CRyToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "ry(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"ry", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CRyToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CRyToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CRyToCXType, CRyToCX);

// quake.ry(θ) target
// ─────────────────────────────────
// quake.phased_rx(θ, π/2) target
struct RyToPhasedRxType;
struct RyToPhasedRx
    : public cudaq::DecompositionPattern<RyToPhasedRxType, quake::RyOp> {
  using cudaq::DecompositionPattern<RyToPhasedRxType,
                                    quake::RyOp>::DecompositionPattern;

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
struct RyToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "ry"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RyToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RyToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RyToPhasedRxType,
                    RyToPhasedRx);

// quake.ry<adj> (θ) target
// ─────────────────────────────────
// quake.ry(-θ) target
struct RyAdjToRyType;
struct RyAdjToRy
    : public cudaq::DecompositionPattern<RyAdjToRyType, quake::RyOp> {
  using cudaq::DecompositionPattern<RyAdjToRyType,
                                    quake::RyOp>::DecompositionPattern;

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
struct RyAdjToRyType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "ry"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"ry"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RyAdjToRy"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RyAdjToRy>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RyAdjToRyType, RyAdjToRy);

//===----------------------------------------------------------------------===//
// RzOp decompositions
//===----------------------------------------------------------------------===//

// quake.rz(λ) [control] target
// ───────────────────────────────
// quake.rz(λ/2) target
// quake.x [control] target
// quake.rz(-λ/2) target
// quake.x [control] target
struct CRzToCXType;
struct CRzToCX : public cudaq::DecompositionPattern<CRzToCXType, quake::RzOp> {
  using cudaq::DecompositionPattern<CRzToCXType,
                                    quake::RzOp>::DecompositionPattern;

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
struct CRzToCXType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rz(1)"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rz", "x(1)"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "CRzToCX"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<CRzToCX>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, CRzToCXType, CRzToCX);

// quake.rz(θ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-θ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct RzToPhasedRxType;
struct RzToPhasedRx
    : public cudaq::DecompositionPattern<RzToPhasedRxType, quake::RzOp> {
  using cudaq::DecompositionPattern<RzToPhasedRxType,
                                    quake::RzOp>::DecompositionPattern;

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
struct RzToPhasedRxType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rz"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"phased_rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RzToPhasedRx"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RzToPhasedRx>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RzToPhasedRxType,
                    RzToPhasedRx);

// quake.rz<adj> (θ) target
// ─────────────────────────────────
// quake.rz(-θ) target
struct RzAdjToRzType;
struct RzAdjToRz
    : public cudaq::DecompositionPattern<RzAdjToRzType, quake::RzOp> {
  using cudaq::DecompositionPattern<RzAdjToRzType,
                                    quake::RzOp>::DecompositionPattern;

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
struct RzAdjToRzType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "rz"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rz"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "RzAdjToRz"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<RzAdjToRz>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, RzAdjToRzType, RzAdjToRz);

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
struct U3ToRotationsType;
struct U3ToRotations
    : public cudaq::DecompositionPattern<U3ToRotationsType, quake::U3Op> {
  using cudaq::DecompositionPattern<U3ToRotationsType,
                                    quake::U3Op>::DecompositionPattern;

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
      // swap the 2nd and 3rd parameter for correctness
      std::swap(phi, lam);
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
struct U3ToRotationsType : public cudaq::DecompositionPatternType {
  using cudaq::DecompositionPatternType::DecompositionPatternType;
  llvm::StringRef getSourceOp() const override { return "u3"; }
  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    static constexpr llvm::StringRef ops[] = {"rz", "rx"};
    return ops;
  }
  llvm::StringRef getPatternName() const override { return "U3ToRotations"; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    std::unique_ptr<mlir::RewritePattern> pattern =
        RewritePattern::create<U3ToRotations>(context, benefit);
    return pattern;
  }
};
CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, U3ToRotationsType,
                    U3ToRotations);

} // namespace

void cudaq::populateWithAllDecompositionPatterns(
    mlir::RewritePatternSet &patterns) {
  for (auto &patternType :
       cudaq::DecompositionPatternType::RegistryType::entries()) {
    patterns.add(patternType.instantiate()->create(patterns.getContext()));
  }
}
