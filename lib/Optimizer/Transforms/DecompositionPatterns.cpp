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
 * 1. The pattern itself, which defines what ops to match and how to replace
 * them. It must inherit from DecompositionPattern<PatternType, Op>.
 * 2. The pattern type, which contains the pattern metadata. It must inherit
 * from DecompositionPatternType.
 * 3. A call to the CUDAQ_REGISTER_TYPE macro to register the pattern in the
 * registry.
 *
 * Writing 2 and 3 manually is a bit verbose. The REGISTER_DECOMPOSITION_PATTERN
 * macro can be used for this purpose instead.
 */

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
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

/// Macro to register a decomposition pattern with its metadata
/// Usage: REGISTER_DECOMPOSITION_PATTERN(PatternName, "source_op", "target1",
/// "target2", ...)
/// where "source_op" is the operation that the pattern matches and
/// {"target1", "target2", ...} are the operations that the pattern may produce.
#define REGISTER_DECOMPOSITION_PATTERN(PATTERN, SOURCE_OP, ...)                \
  struct PATTERN##Type : public cudaq::DecompositionPatternType {              \
    using cudaq::DecompositionPatternType::DecompositionPatternType;           \
    llvm::StringRef getSourceOp() const override { return SOURCE_OP; }         \
    llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {            \
      static constexpr llvm::StringRef ops[] = {__VA_ARGS__};                  \
      return ops;                                                              \
    }                                                                          \
    llvm::StringRef getPatternName() const override { return #PATTERN; }       \
    std::unique_ptr<mlir::RewritePattern>                                      \
    create(mlir::MLIRContext *context,                                         \
           mlir::PatternBenefit benefit = 1) const override {                  \
      std::unique_ptr<mlir::RewritePattern> pattern =                          \
          RewritePattern::create<PATTERN>(context, benefit);                   \
      return pattern;                                                          \
    }                                                                          \
  };                                                                           \
  CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, PATTERN##Type, PATTERN)

// TODO: The decomposition patterns "SToR1", "TToR1", "R1ToU3", "U3ToRotations"
// can handle arbitrary number of controls, but currently metadata cannot
// capture this. The pattern types therefore only advertise them for 0 controls.

//===----------------------------------------------------------------------===//
// HOp decompositions
//===----------------------------------------------------------------------===//

// quake.h target
// ───────────────────────────────────
// quake.phased_rx(π/2, π/2) target
// quake.phased_rx(π, 0) target

struct HToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(HToPhasedRx, "h", "phased_rx");

// quake.exp_pauli(theta) target pauliWord
// ───────────────────────────────────
// Basis change operations, cnots, rz(theta), adjoint basis change
struct ExpPauliDecompositionType; // forward declare the pattern type, defined
                                  // in the macro below
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
REGISTER_DECOMPOSITION_PATTERN(ExpPauliDecomposition, "exp_pauli", "rx", "h",
                               "x(1)", "rz");

// Naive mapping of R1 to Rz, ignoring the global phase.
// This is only expected to work with full inlining and
// quake apply specialization.
struct R1ToRzType; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(R1ToRz, "r1", "rz");

// Naive mapping of R1 to U3
// quake.r1(λ) [control] target
// ───────────────────────────────────
// quake.u3(0, 0, λ) [control] target
struct R1ToU3Type; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(R1ToU3, "r1", "u3");

// quake.r1<adj> (θ) target
// ─────────────────────────────────
// quake.r1(-θ) target
struct R1AdjToR1Type; // forward declare the pattern type, defined in the macro
                      // below
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
REGISTER_DECOMPOSITION_PATTERN(R1AdjToR1, "r1", "r1");

// quake.swap a, b
// ───────────────────────────────────
// quake.cnot b, a;
// quake.cnot a, b;
// quake.cnot b, a;
struct SwapToCXType; // forward declare the pattern type, defined in the macro
                     // below
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
REGISTER_DECOMPOSITION_PATTERN(SwapToCX, "swap", "x(1)");

// quake.h control, target
// ───────────────────────────────────
// quake.s target;
// quake.h target;
// quake.t target;
// quake.x control, target;
// quake.t<adj> target;
// quake.h target;
// quake.s<adj> target;
struct CHToCXType; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(CHToCX, "h(1)", "s", "h", "t", "x(1)");

//===----------------------------------------------------------------------===//
// SOp decompositions
//===----------------------------------------------------------------------===//

// quake.s target
// ──────────────────────────────
// phased_rx(π/2, 0) target
// phased_rx(-π/2, π/2) target
// phased_rx(-π/2, 0) target
struct SToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(SToPhasedRx, "s", "phased_rx");

// quake.s [control] target
// ────────────────────────────────────
// quake.r1(π/2) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct SToR1Type; // forward declare the pattern type, defined in the macro
                  // below
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
REGISTER_DECOMPOSITION_PATTERN(SToR1, "s", "r1");

//===----------------------------------------------------------------------===//
// TOp decompositions
//===----------------------------------------------------------------------===//

// quake.t target
// ────────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π/4, π/2) target
// quake.phased_rx(-π/2, 0) target
struct TToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(TToPhasedRx, "t", "phased_rx");

// quake.t [control] target
// ────────────────────────────────────
// quake.r1(π/4) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct TToR1Type; // forward declare the pattern type, defined in the macro
                  // below
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
REGISTER_DECOMPOSITION_PATTERN(TToR1, "t", "r1");

//===----------------------------------------------------------------------===//
// XOp decompositions
//===----------------------------------------------------------------------===//

// quake.x [control] target
// ──────────────────────────────────
// quake.h target
// quake.z [control] target
// quake.h target
struct CXToCZType; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(CXToCZ, "x(1)", "h", "z(1)");

// quake.x [controls] target
// ──────────────────────────────────
// quake.h target
// quake.z [controls] target
// quake.h target
struct CCXToCCZType; // forward declare the pattern type, defined in the macro
                     // below
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
REGISTER_DECOMPOSITION_PATTERN(CCXToCCZ, "x(2)", "h", "z(2)");

// quake.x target
// ───────────────────────────────
// quake.phased_rx(π, 0) target
struct XToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(XToPhasedRx, "x", "phased_rx");

//===----------------------------------------------------------------------===//
// YOp decompositions
//===----------------------------------------------------------------------===//

// quake.y target
// ─────────────────────────────────
// quake.phased_rx(π, -π/2) target
struct YToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(YToPhasedRx, "y", "phased_rx");

// quake.y [control] target
// ───────────────────────────────────
// quake.s<adj> target;
// quake.x [control] target;
// quake.s target;

struct CYToCXType; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(CYToCX, "y(1)", "s", "x(1)");

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
struct CCZToCXType; // forward declare the pattern type, defined in the macro
                    // below
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
REGISTER_DECOMPOSITION_PATTERN(CCZToCX, "z(2)", "t", "x(1)");

// quake.z [control] target
// ──────────────────────────────────
// quake.h target
// quake.x [control] target
// quake.h target

struct CZToCXType; // forward declare the pattern type, defined in the macro
                   // below
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
REGISTER_DECOMPOSITION_PATTERN(CZToCX, "z(1)", "h", "x(1)");

// quake.z target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π, π/2) target
// quake.phased_rx(-π/2, 0) target
struct ZToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
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
REGISTER_DECOMPOSITION_PATTERN(ZToPhasedRx, "z", "phased_rx");

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
struct CR1ToCXType; // forward declare the pattern type, defined in the macro
                    // below
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
REGISTER_DECOMPOSITION_PATTERN(CR1ToCX, "r1(1)", "r1", "x(1)");

// quake.r1(λ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-λ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct R1ToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
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
REGISTER_DECOMPOSITION_PATTERN(R1ToPhasedRx, "r1", "phased_rx");

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
struct CRxToCXType; // forward declare the pattern type, defined in the macro
                    // below
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
REGISTER_DECOMPOSITION_PATTERN(CRxToCX, "rx(1)", "s", "x(1)", "ry", "rz");

// quake.rx(θ) target
// ───────────────────────────────
// quake.phased_rx(θ, 0) target
struct RxToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
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
REGISTER_DECOMPOSITION_PATTERN(RxToPhasedRx, "rx", "phased_rx");

// quake.rx<adj> (θ) target
// ─────────────────────────────────
// quake.rx(-θ) target
struct RxAdjToRxType; // forward declare the pattern type, defined in the macro
                      // below
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
REGISTER_DECOMPOSITION_PATTERN(RxAdjToRx, "rx", "rx");

//===----------------------------------------------------------------------===//
// RyOp decompositions
//===----------------------------------------------------------------------===//

// quake.ry(θ) [control] target
// ───────────────────────────────
// quake.ry(θ/2) target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
struct CRyToCXType; // forward declare the pattern type, defined in the macro
                    // below
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
REGISTER_DECOMPOSITION_PATTERN(CRyToCX, "ry(1)", "ry", "x(1)");

// quake.ry(θ) target
// ─────────────────────────────────
// quake.phased_rx(θ, π/2) target
struct RyToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
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
REGISTER_DECOMPOSITION_PATTERN(RyToPhasedRx, "ry", "phased_rx");

// quake.ry<adj> (θ) target
// ─────────────────────────────────
// quake.ry(-θ) target
struct RyAdjToRyType; // forward declare the pattern type, defined in the macro
                      // below
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
REGISTER_DECOMPOSITION_PATTERN(RyAdjToRy, "ry", "ry");

//===----------------------------------------------------------------------===//
// RzOp decompositions
//===----------------------------------------------------------------------===//

// quake.rz(λ) [control] target
// ───────────────────────────────
// quake.rz(λ/2) target
// quake.x [control] target
// quake.rz(-λ/2) target
// quake.x [control] target
struct CRzToCXType; // forward declare the pattern type, defined in the macro
                    // below
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
REGISTER_DECOMPOSITION_PATTERN(CRzToCX, "rz(1)", "rz", "x(1)");

// quake.rz(θ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-θ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct RzToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
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
REGISTER_DECOMPOSITION_PATTERN(RzToPhasedRx, "rz", "phased_rx");

// quake.rz<adj> (θ) target
// ─────────────────────────────────
// quake.rz(-θ) target
struct RzAdjToRzType; // forward declare the pattern type, defined in the macro
                      // below
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
REGISTER_DECOMPOSITION_PATTERN(RzAdjToRz, "rz", "rz");

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
struct U3ToRotationsType; // forward declare the pattern type, defined in the
                          // macro below
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
REGISTER_DECOMPOSITION_PATTERN(U3ToRotations, "u3", "rz", "rx");

} // namespace

void cudaq::populateWithAllDecompositionPatterns(
    mlir::RewritePatternSet &patterns) {
  // For deterministic ordering, sort the registered pattern types by name
  // Note that this assumes that no additional patterns are registered at
  // runtime.
  static std::map<std::string, std::unique_ptr<cudaq::DecompositionPatternType>>
      patternTypes = []() {
        std::map<std::string, std::unique_ptr<cudaq::DecompositionPatternType>>
            map;
        for (auto &patternType :
             cudaq::DecompositionPatternType::RegistryType::entries()) {
          map[patternType.getName().str()] = patternType.instantiate();
        }
        return map;
      }();

  for (auto it = patternTypes.begin(), ie = patternTypes.end(); it != ie;
       ++it) {
    patterns.add(it->second->create(patterns.getContext()));
  }
}
