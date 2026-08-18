/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "QuakeOperatorCreator.h"
#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TypeName.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <optional>

/**
 * This file contains the decomposition patterns that match single gates and
 * decompose them into a sequence of other gates.
 *
 * Each pattern definition contains 3 elements:
 * 1. The pattern itself, which defines what ops to match and how to replace
 * them. It must inherit from DecompositionPattern<PatternType, Op>.
 * 2. A call to the REGISTER_DECOMPOSITION_PATTERN macro to register the pattern
 * in the registry and define its metadata.
 */

using namespace mlir;

LLVM_INSTANTIATE_REGISTRY(cudaq::DecompositionPatternTypeRegistry)

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
  return arith::ConstantIntOp::create(rewriter, loc, value, 64);
}

inline Value createDivF(Location loc, Value numerator, double denominator,
                        PatternRewriter &rewriter) {
  auto denominatorValue =
      createConstant(loc, denominator, numerator.getType(), rewriter);
  return arith::DivFOp::create(rewriter, loc, numerator, denominatorValue);
}

/// @brief Returns true if \p op contains any `ControlType` operands.
inline bool containsControlTypes(cudaq::quake::OperatorInterface op) {
  return llvm::any_of(op.getControls(), [](const Value &v) {
    return isa<cudaq::quake::ControlType>(v.getType());
  });
}

std::optional<std::size_t>
cudaq::getKnownNumControls(cudaq::quake::OperatorInterface op) {
  std::size_t numControls = 0;
  for (auto control : op.getControls()) {
    if (auto veq = dyn_cast<cudaq::quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return std::nullopt;
      numControls += veq.getSize();
      continue;
    }
    numControls += 1;
  }
  return numControls;
}

/// Check whether the operation has the correct number of controls.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
static LogicalResult checkNumControls(cudaq::quake::OperatorInterface op,
                                      std::size_t requiredNumControls) {
  if (op.getControls().size() > requiredNumControls)
    return failure();

  auto numControls = cudaq::getKnownNumControls(op);
  return numControls && *numControls == requiredNumControls ? success()
                                                            : failure();
}

/// Check whether the operation has the correct number of controls. This
/// function take as input a mutable array reference, `controls`, which must
/// have the size equal to the number of controls. If the operation has `veq`s
/// as controls, split those into single qubit references.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
static LogicalResult checkAndExtractControls(cudaq::quake::OperatorInterface op,
                                             MutableArrayRef<Value> controls,
                                             PatternRewriter &rewriter) {
  if (failed(checkNumControls(op, controls.size())))
    return failure();

  std::size_t controlIndex = 0;
  for (Value control : op.getControls()) {
    if (auto veq = dyn_cast<cudaq::quake::VeqType>(control.getType())) {
      for (std::size_t i = 0, end = veq.getSize(); i < end; ++i) {
        Value index = createConstant(op.getLoc(), i, rewriter);
        Value qref = cudaq::quake::ExtractRefOp::create(rewriter, op.getLoc(),
                                                        control, index);
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
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

/// Macro helper to register a decomposition pattern with its metadata.
///
/// A pattern is composed of one or multiple variants. Each argument after the
/// pattern name is one variant, written as a brace-enclosed list where the
/// first element is the source op and the remaining elements are the target
/// ops.
///
/// Single variant:
/// ```
/// REGISTER_DECOMPOSITION_PATTERN(PatternName, {"source_op", "target1",
/// "target2"})
/// ```
/// Multiple variants:
/// ```
/// REGISTER_DECOMPOSITION_PATTERN(PatternName, {"source_op",    "target1"},
/// {"source_op(1)", "target1(1)"})
/// ```
#undef REGISTER_DECOMPOSITION_PATTERN
#define REGISTER_DECOMPOSITION_PATTERN(PATTERN, ...)                           \
  struct PATTERN##Type : public cudaq::DecompositionPatternType {              \
    PATTERN##Type()                                                            \
        : cudaq::DecompositionPatternType(                                     \
              std::vector<cudaq::DecompositionPatternVariant>{__VA_ARGS__}) {} \
    llvm::StringRef getPatternName() const override { return #PATTERN; }       \
    std::unique_ptr<mlir::RewritePattern>                                      \
    create(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1,       \
           llvm::ArrayRef<std::size_t> disabledCtrlCnts = {}) const override { \
      std::unique_ptr<mlir::RewritePattern> pattern =                          \
          RewritePattern::create<PATTERN>(context, benefit, disabledCtrlCnts); \
      return pattern;                                                          \
    }                                                                          \
  };                                                                           \
  static cudaq::DecompositionPatternTypeRegistry::Add<PATTERN##Type> CONCAT(   \
      TEMPNAME_, PATTERN)(#PATTERN, "");

// Bare S/T must stay separate from controlled S/T so targets can preserve
// bare Clifford+T gates without accepting all controlled forms.

//===----------------------------------------------------------------------===//
// HOp decompositions
//===----------------------------------------------------------------------===//

namespace {
using cudaq::opt::decomp::QuakeOperatorCreator;

// quake.h target
// ───────────────────────────────────
// quake.phased_rx(π/2, π/2) target
// quake.phased_rx(π, 0) target
struct HToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
struct HToPhasedRx
    : public cudaq::DecompositionPattern<HToPhasedRxType, cudaq::quake::HOp> {

  using cudaq::DecompositionPattern<HToPhasedRxType,
                                    cudaq::quake::HOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::HOp op,
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
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = pi;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(HToPhasedRx, {"h", "phased_rx"});

// quake.exp_pauli(theta) target pauliWord
// ───────────────────────────────────
// Basis change operations, cnots, rz(theta), adjoint basis change
struct ExpPauliDecompositionType; // forward declare the pattern type, defined
                                  // in the macro below
struct ExpPauliDecomposition
    : public cudaq::DecompositionPattern<ExpPauliDecompositionType,
                                         cudaq::quake::ExpPauliOp> {
  using cudaq::DecompositionPattern<
      ExpPauliDecompositionType,
      cudaq::quake::ExpPauliOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExpPauliOp expPauliOp,
                                PatternRewriter &rewriter) const override {
    auto loc = expPauliOp.getLoc();
    auto module = expPauliOp->getParentOfType<ModuleOp>();
    auto theta = expPauliOp.getParameter();
    auto pauliWord = expPauliOp.getPauli();

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
                  pauliWord.getDefiningOp<cudaq::cc::SequenceInitOp>()) {
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
              // sequence structure.
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

    auto maybePaulis = cudaq::quake::symbolizePauliWord(
        StringRef(pauliWordStr).take_front(size));
    if (!maybePaulis)
      return expPauliOp.emitOpError(
          "Pauli word must contain only I, X, Y, or Z");
    const auto &paulis = *maybePaulis;

    // Determine target cardinalities before creating any IR. Pattern failure
    // must leave the operation unchanged so that the greedy driver can stop
    // cleanly instead of repeatedly matching a partially rewritten operation.
    SmallVector<std::size_t> targetSizes;
    auto targets = expPauliOp.getTargets();
    std::size_t qubitCount = 0;
    for (Value target : targets) {
      auto targetTy = target.getType();
      if (isa<cudaq::quake::RefType>(targetTy)) {
        if (qubitCount == paulis.size())
          return expPauliOp.emitOpError(
              "Pauli word length must match target qubit count");
        targetSizes.push_back(1);
        ++qubitCount;
        continue;
      }
      if (!isa<cudaq::quake::VeqType>(targetTy))
        return failure();
      auto maybeSize = cudaq::quake::getVeqSize(target);
      if (!maybeSize) {
        // The Pauli-word length cannot determine boundaries between dynamic
        // targets.
        if (targets.size() != 1)
          return failure();
        maybeSize = paulis.size();
      }
      if (*maybeSize > paulis.size() - qubitCount)
        return expPauliOp.emitOpError(
            "Pauli word length must match target qubit count");
      targetSizes.push_back(*maybeSize);
      qubitCount += *maybeSize;
    }

    if (qubitCount != paulis.size())
      return expPauliOp.emitOpError(
          "Pauli word length must match target qubit count");

    // Flatten variadic targets into individual refs before lowering.
    SmallVector<Value> qubits;
    for (auto [target, targetSize] : llvm::zip(targets, targetSizes)) {
      if (isa<cudaq::quake::RefType>(target.getType())) {
        qubits.push_back(target);
        continue;
      }
      for (std::size_t i = 0; i < targetSize; ++i) {
        Value index = arith::ConstantIntOp::create(rewriter, loc, i, 64);
        qubits.push_back(
            cudaq::quake::ExtractRefOp::create(rewriter, loc, target, index));
      }
    }

    if (expPauliOp.isAdj())
      theta = arith::NegFOp::create(rewriter, loc, theta);

    SmallVector<Value> qubitSupport;
    for (auto [i, pauli] : llvm::enumerate(paulis)) {
      Value qubitI = qubits[i];
      if (pauli != cudaq::quake::Pauli::I)
        qubitSupport.push_back(qubitI);

      if (pauli == cudaq::quake::Pauli::Y) {
        APFloat d(M_PI_2);
        Value param = arith::ConstantFloatOp::create(rewriter, loc,
                                                     rewriter.getF64Type(), d);
        cudaq::quake::RxOp::create(rewriter, loc, ValueRange{param},
                                   ValueRange{}, ValueRange{qubitI});
      } else if (pauli == cudaq::quake::Pauli::X) {
        cudaq::quake::HOp::create(rewriter, loc, ValueRange{qubitI});
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
      cudaq::quake::XOp::create(rewriter, loc, ValueRange{qubitSupport[i]},
                                ValueRange{qubitSupport[i + 1]});
      toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
    }

    // Note: `Rz(theta)` = `exp(-i*theta/2 Z)`
    Value negTwoTheta = arith::MulFOp::create(
        rewriter, loc,
        createConstant(loc, -2.0, rewriter.getF64Type(), rewriter), theta);
    cudaq::quake::RzOp::create(rewriter, loc, ValueRange{negTwoTheta},
                               ValueRange{}, ValueRange{qubitSupport.back()});

    std::reverse(toReverse.begin(), toReverse.end());
    for (auto &[i, j] : toReverse)
      cudaq::quake::XOp::create(rewriter, loc, ValueRange{i}, ValueRange{j});

    for (std::size_t i = 0; i < paulis.size(); i++) {
      std::size_t k = paulis.size() - 1 - i;
      Value qubitK = qubits[k];

      if (paulis[k] == cudaq::quake::Pauli::Y) {
        APFloat d(-M_PI_2);
        Value param = arith::ConstantFloatOp::create(rewriter, loc,
                                                     rewriter.getF64Type(), d);
        cudaq::quake::RxOp::create(rewriter, loc, ValueRange{param},
                                   ValueRange{}, ValueRange{qubitK});
      } else if (paulis[k] == cudaq::quake::Pauli::X) {
        cudaq::quake::HOp::create(rewriter, loc, ValueRange{qubitK});
      }
    }

    rewriter.eraseOp(expPauliOp);

    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(ExpPauliDecomposition,
                               {"exp_pauli", "rx", "h", "x(1)", "rz"});

// Exact mapping of R1 to Rz plus its phase residue.
struct R1ToRzType; // forward declare the pattern type, defined in the macro
                   // below
struct R1ToRz
    : public cudaq::DecompositionPattern<R1ToRzType, cudaq::quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToRzType,
                                    cudaq::quake::R1Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    if (!isEnabled(cudaq::getKnownNumControls(r1Op)))
      return failure();

    Location location = r1Op.getLoc();
    SmallVector<Value> controls(r1Op.getControls());
    SmallVector<Value> targets(r1Op.getTargets());

    // PhaseOp requires a scalar anchor. Do not partially rewrite an aggregate
    // R1 because selecting one arbitrary `veq` element would represent the
    // wrong phase for the _aggregate_ operation.

    if (targets.empty() || !isa<cudaq::quake::RefType, cudaq::quake::WireType>(
                               targets.back().getType()))
      return rewriter.notifyMatchFailure(
          r1Op,
          "R1ToRz requires a scalar target to anchor its phase correction");

    auto resultTypes =
        cudaq::opt::getWireResultTypes(rewriter, controls, targets);
    auto rz = cudaq::quake::RzOp::create(
        rewriter, location, resultTypes, r1Op.getIsAdjAttr(),
        r1Op.getParameters(), controls, targets,
        r1Op.getNegatedQubitControlsAttr());
    cudaq::opt::threadWireResults(rz, controls, targets);

    // Preserve a literal zero so emitPhaseCorrection can omit the correction.
    // Constructing a `0 / 2` first would hide the zero behind an arith.divf
    // until a later canonicalization pass.
    Value phase = r1Op.getParameter();
    if (!matchPattern(phase, m_AnyZeroFloat())) {
      phase = createDivF(location, phase, 2.0, rewriter);
      if (r1Op.isAdj())
        phase = arith::NegFOp::create(rewriter, location, phase);
    }

    auto correction = cudaq::opt::emitPhaseCorrection(
        rewriter, location, phase, controls, r1Op.getNegatedQubitControlsAttr(),
        targets.back());
    controls = std::move(correction.controls);
    targets.back() = correction.anchor;

    rewriter.replaceOp(r1Op, cudaq::opt::getWireValues(controls, targets));
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(R1ToRz, {"r1", "rz"}, {"r1(n)", "rz(n)"});

// Naive mapping of R1 to U3
// quake.r1(λ) [control] target
// ───────────────────────────────────
// quake.u3(0, 0, λ) [control] target
struct R1ToU3Type; // forward declare the pattern type, defined in the macro
                   // below
struct R1ToU3
    : public cudaq::DecompositionPattern<R1ToU3Type, cudaq::quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToU3Type,
                                    cudaq::quake::R1Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    // Dijkstra selects concrete source variants such as r1(1) from this
    // pattern's r1(n) metadata. Only rewrite variants selected for this pass.
    if (!isEnabled(cudaq::getKnownNumControls(r1Op)))
      return failure();

    Location loc = r1Op->getLoc();
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    std::array<Value, 3> parameters = {zero, zero, r1Op.getParameters()[0]};
    rewriter.replaceOpWithNewOp<cudaq::quake::U3Op>(
        r1Op, r1Op.isAdj(), parameters, r1Op.getControls(), r1Op.getTargets());
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(R1ToU3, {"r1(n)", "u3(n)"});

// quake.r1<adj> (θ) target
// ─────────────────────────────────
// quake.r1(-θ) target
struct R1AdjToR1Type; // forward declare the pattern type, defined in the macro
                      // below
struct R1AdjToR1
    : public cudaq::DecompositionPattern<R1AdjToR1Type, cudaq::quake::R1Op> {
  using cudaq::DecompositionPattern<R1AdjToR1Type,
                                    cudaq::quake::R1Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::R1Op>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(R1AdjToR1, {"r1<adj>", "r1"});

// quake.swap a, b
// ───────────────────────────────────
// quake.cnot b, a;
// quake.cnot a, b;
// quake.cnot b, a;
struct SwapToCXType; // forward declare the pattern type, defined in the macro
                     // below
struct SwapToCX
    : public cudaq::DecompositionPattern<SwapToCXType, cudaq::quake::SwapOp> {
  using cudaq::DecompositionPattern<SwapToCXType,
                                    cudaq::quake::SwapOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::SwapOp op,
                                PatternRewriter &rewriter) const override {
    // Op info
    Location loc = op->getLoc();
    Value a = op.getTarget(0);
    Value b = op.getTarget(1);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::XOp>(loc, b, a);
    qRewriter.create<cudaq::quake::XOp>(loc, a, b);
    qRewriter.create<cudaq::quake::XOp>(loc, b, a);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{a, b});
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(SwapToCX, {"swap", "x(1)"});

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
struct CHToCX
    : public cudaq::DecompositionPattern<CHToCXType, cudaq::quake::HOp> {
  using cudaq::DecompositionPattern<CHToCXType,
                                    cudaq::quake::HOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::HOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value control = op.getControls()[0];
    Value target = op.getTarget();

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::SOp>(loc, target);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    qRewriter.create<cudaq::quake::TOp>(loc, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/true, target);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    qRewriter.create<cudaq::quake::SOp>(loc, /*isAdj=*/true, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};
// TODO: Technically, this pattern also produces s<adj> and t<adj> ops, but we
// currently don't treat them as distinct from their non-adjoint counterparts.
REGISTER_DECOMPOSITION_PATTERN(CHToCX, {"h(1)", "s", "h", "t", "x(1)"});

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
    : public cudaq::DecompositionPattern<SToPhasedRxType, cudaq::quake::SOp> {
  using cudaq::DecompositionPattern<SToPhasedRxType,
                                    cudaq::quake::SOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::SOp op,
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
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);

    Value angle = op.isAdj() ? pi_2 : negPi_2;

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(SToPhasedRx, {"s", "phased_rx"});

// quake.s [controls] target
// ────────────────────────────────────
// quake.r1(π/2) [controls] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct SToR1Type; // forward declare the pattern type, defined in the macro
                  // below
struct SToR1
    : public cudaq::DecompositionPattern<SToR1Type, cudaq::quake::SOp> {
  using cudaq::DecompositionPattern<SToR1Type,
                                    cudaq::quake::SOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::SOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<std::size_t> numControls = cudaq::getKnownNumControls(op);
    if (!isEnabled(numControls))
      return failure();

    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_2 : M_PI_2,
                                rewriter.getF64Type(), rewriter);

    SmallVector<Value> controls(op.getControls());
    Value target = op.getTarget();
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::R1Op>(loc, angle, controls, target);

    if (numControls.has_value() && *numControls == 0)
      qRewriter.selectWiresAndReplaceUses(op, target);
    else
      qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(SToR1, {"s", "r1"}, {"s(1)", "r1(1)"},
                               {"s(n)", "r1(n)"});

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
    : public cudaq::DecompositionPattern<TToPhasedRxType, cudaq::quake::TOp> {
  using cudaq::DecompositionPattern<TToPhasedRxType,
                                    cudaq::quake::TOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::TOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = createConstant(loc, -M_PI_4, rewriter.getF64Type(), rewriter);
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(TToPhasedRx, {"t", "phased_rx"});

// quake.t [controls] target
// ────────────────────────────────────
// quake.r1(π/4) [controls] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct TToR1Type; // forward declare the pattern type, defined in the macro
                  // below
struct TToR1
    : public cudaq::DecompositionPattern<TToR1Type, cudaq::quake::TOp> {
  using cudaq::DecompositionPattern<TToR1Type,
                                    cudaq::quake::TOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::TOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<std::size_t> numControls = cudaq::getKnownNumControls(op);
    if (!isEnabled(numControls))
      return failure();

    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_4 : M_PI_4,
                                rewriter.getF64Type(), rewriter);
    SmallVector<Value> controls(op.getControls());
    Value target = op.getTarget();
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::R1Op>(loc, angle, controls, target);

    if (numControls.has_value() && *numControls == 0)
      qRewriter.selectWiresAndReplaceUses(op, target);
    else
      qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(TToR1, {"t", "r1"}, {"t(1)", "r1(1)"},
                               {"t(n)", "r1(n)"});

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
struct CXToCZ
    : public cudaq::DecompositionPattern<CXToCZType, cudaq::quake::XOp> {
  using cudaq::DecompositionPattern<CXToCZType,
                                    cudaq::quake::XOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::XOp op,
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
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    if (negControl)
      qRewriter.create<cudaq::quake::XOp>(loc, controls);
    qRewriter.create<cudaq::quake::ZOp>(loc, controls, target);
    if (negControl)
      qRewriter.create<cudaq::quake::XOp>(loc, controls);
    qRewriter.create<cudaq::quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CXToCZ, {"x(1)", "h", "z(1)"});

// quake.x [controls] target
// ──────────────────────────────────
// quake.h target
// quake.z [controls] target
// quake.h target
struct CCXToCCZType; // forward declare the pattern type, defined in the macro
                     // below
struct CCXToCCZ
    : public cudaq::DecompositionPattern<CCXToCCZType, cudaq::quake::XOp> {
  using cudaq::DecompositionPattern<CCXToCCZType,
                                    cudaq::quake::XOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::XOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 2> controls(2);
    if (failed(checkAndExtractControls(op, controls, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    auto zOp = qRewriter.create<cudaq::quake::ZOp>(loc, controls, target);
    zOp.setNegatedQubitControls(op.getNegatedQubitControls());
    qRewriter.create<cudaq::quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CCXToCCZ, {"x(2)", "h", "z(2)"});

// quake.x target
// ───────────────────────────────
// quake.phased_rx(π, 0) target
struct XToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
struct XToPhasedRx
    : public cudaq::DecompositionPattern<XToPhasedRxType, cudaq::quake::XOp> {
  using cudaq::DecompositionPattern<XToPhasedRxType,
                                    cudaq::quake::XOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::XOp op,
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
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(XToPhasedRx, {"x", "phased_rx"});

//===----------------------------------------------------------------------===//
// YOp decompositions
//===----------------------------------------------------------------------===//

// quake.y target
// ─────────────────────────────────
// quake.phased_rx(π, -π/2) target
struct YToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
struct YToPhasedRx
    : public cudaq::DecompositionPattern<YToPhasedRxType, cudaq::quake::YOp> {
  using cudaq::DecompositionPattern<YToPhasedRxType,
                                    cudaq::quake::YOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::YOp op,
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
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(YToPhasedRx, {"y", "phased_rx"});

// quake.y [control] target
// ───────────────────────────────────
// quake.s<adj> target;
// quake.x [control] target;
// quake.s target;

struct CYToCXType; // forward declare the pattern type, defined in the macro
                   // below
struct CYToCX
    : public cudaq::DecompositionPattern<CYToCXType, cudaq::quake::YOp> {
  using cudaq::DecompositionPattern<CYToCXType,
                                    cudaq::quake::YOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::YOp op,
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
    qRewriter.create<cudaq::quake::SOp>(loc, /*isAdj=*/true, target);
    qRewriter.create<cudaq::quake::XOp>(loc, controls, target);
    qRewriter.create<cudaq::quake::SOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
// TODO: Technically, this pattern also produces s<adj> ops, but we currently
// don't treat it as distinct from their non-adjoint counterparts.
REGISTER_DECOMPOSITION_PATTERN(CYToCX, {"y(1)", "s", "x(1)"});

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
struct CCZToCX
    : public cudaq::DecompositionPattern<CCZToCXType, cudaq::quake::ZOp> {
  using cudaq::DecompositionPattern<CCZToCXType,
                                    cudaq::quake::ZOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::ZOp op,
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
    qRewriter.create<cudaq::quake::XOp>(loc, controls[1], target);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/!negC0, target);
    qRewriter.create<cudaq::quake::XOp>(loc, controls[0], target);
    qRewriter.create<cudaq::quake::TOp>(loc, target);
    qRewriter.create<cudaq::quake::XOp>(loc, controls[1], target);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/!negC1, target);
    qRewriter.create<cudaq::quake::XOp>(loc, controls[0], target);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/negC0 && !negC1, target);

    qRewriter.create<cudaq::quake::XOp>(loc, controls[0], controls[1]);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/true, controls[1]);
    qRewriter.create<cudaq::quake::XOp>(loc, controls[0], controls[1]);
    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/negC0, controls[1]);

    qRewriter.create<cudaq::quake::TOp>(loc, /*isAdj=*/negC1, controls[0]);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
// TODO: Technically, this pattern also produces t<adj> ops, but we currently
// don't treat it as distinct from their non-adjoint counterparts.
REGISTER_DECOMPOSITION_PATTERN(CCZToCX, {"z(2)", "t", "x(1)"});

// quake.z [control] target
// ──────────────────────────────────
// quake.h target
// quake.x [control] target
// quake.h target

struct CZToCXType; // forward declare the pattern type, defined in the macro
                   // below
struct CZToCX
    : public cudaq::DecompositionPattern<CZToCXType, cudaq::quake::ZOp> {
  using cudaq::DecompositionPattern<CZToCXType,
                                    cudaq::quake::ZOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::ZOp op,
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
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    if (negControl)
      qRewriter.create<cudaq::quake::XOp>(loc, controls);
    qRewriter.create<cudaq::quake::XOp>(loc, controls, target);
    if (negControl)
      qRewriter.create<cudaq::quake::XOp>(loc, controls);
    qRewriter.create<cudaq::quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CZToCX, {"z(1)", "h", "x(1)"});

// quake.z target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π, π/2) target
// quake.phased_rx(-π/2, 0) target
struct ZToPhasedRxType; // forward declare the pattern type, defined in the
                        // macro below
struct ZToPhasedRx
    : public cudaq::DecompositionPattern<ZToPhasedRxType, cudaq::quake::ZOp> {
  using cudaq::DecompositionPattern<ZToPhasedRxType,
                                    cudaq::quake::ZOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::ZOp op,
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
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi;
    parameters[1] = pi_2;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(ZToPhasedRx, {"z", "phased_rx"});

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
struct CR1ToCX
    : public cudaq::DecompositionPattern<CR1ToCXType, cudaq::quake::R1Op> {
  using cudaq::DecompositionPattern<CR1ToCXType,
                                    cudaq::quake::R1Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::R1Op op,
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
      angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = arith::NegFOp::create(rewriter, loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::R1Op>(loc, /*isAdj*/ negControl, halfAngle,
                                         noControls, control);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::R1Op>(loc, /*isAdj*/ negControl,
                                         negHalfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::R1Op>(loc, halfAngle, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CR1ToCX, {"r1(1)", "r1", "x(1)"});

// quake.r1(λ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-λ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct R1ToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
struct R1ToPhasedRx
    : public cudaq::DecompositionPattern<R1ToPhasedRxType, cudaq::quake::R1Op> {
  using cudaq::DecompositionPattern<R1ToPhasedRxType,
                                    cudaq::quake::R1Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);
    Value negAngle = arith::NegFOp::create(rewriter, loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(R1ToPhasedRx, {"r1", "phased_rx"});

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
struct CRxToCX
    : public cudaq::DecompositionPattern<CRxToCXType, cudaq::quake::RxOp> {
  using cudaq::DecompositionPattern<CRxToCXType,
                                    cudaq::quake::RxOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RxOp op,
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
      angle = arith::NegFOp::create(rewriter, loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = arith::NegFOp::create(rewriter, loc, halfAngle);
    Value negPI_2 = createConstant(loc, -M_PI_2, angleType, rewriter);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::SOp>(loc, /*isAdj*/ negControl, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::RyOp>(loc, negHalfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::RyOp>(loc, /*isAdj*/ negControl, halfAngle,
                                         noControls, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, /*isAdj*/ negControl, negPI_2,
                                         noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CRxToCX, {"rx(1)", "s", "x(1)", "ry", "rz"});

// quake.rx(θ) target
// ───────────────────────────────
// quake.phased_rx(θ, 0) target
struct RxToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
struct RxToPhasedRx
    : public cudaq::DecompositionPattern<RxToPhasedRxType, cudaq::quake::RxOp> {
  using cudaq::DecompositionPattern<RxToPhasedRxType,
                                    cudaq::quake::RxOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);

    SmallVector<Value> parameters = {angle, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RxToPhasedRx, {"rx", "phased_rx"});

// quake.rx(θ) target
// ───────────────────────────────
// quake.h target
// quake.rz(θ) target
// quake.h target
//
// Exact identity Rx(θ) = H . Rz(θ) . H (since H X H = Z). Gives passes a way
// to reach an Rz+Clifford basis. It is used ahead of clifford-t-synthesis so
// that synthesis only has to handle Rz.
struct RxToRzType; // forward declare the pattern type, defined in the macro
                   // below
struct RxToRz
    : public cudaq::DecompositionPattern<RxToRzType, cudaq::quake::RxOp> {
  using cudaq::DecompositionPattern<RxToRzType,
                                    cudaq::quake::RxOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);

    SmallVector<Value> noControls;
    SmallVector<Value> rzParams = {angle};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, rzParams, noControls, target);
    qRewriter.create<cudaq::quake::HOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RxToRz, {"rx", "h", "rz"});

// quake.rx<adj> (θ) target
// ─────────────────────────────────
// quake.rx(-θ) target
struct RxAdjToRxType; // forward declare the pattern type, defined in the macro
                      // below
struct RxAdjToRx
    : public cudaq::DecompositionPattern<RxAdjToRxType, cudaq::quake::RxOp> {
  using cudaq::DecompositionPattern<RxAdjToRxType,
                                    cudaq::quake::RxOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RxOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RxAdjToRx, {"rx<adj>", "rx"});
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
struct CRyToCX
    : public cudaq::DecompositionPattern<CRyToCXType, cudaq::quake::RyOp> {
  using cudaq::DecompositionPattern<CRyToCXType,
                                    cudaq::quake::RyOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RyOp op,
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
      angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = arith::NegFOp::create(rewriter, loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RyOp>(loc, halfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::RyOp>(loc, /*isAdj*/ negControl,
                                         negHalfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CRyToCX, {"ry(1)", "ry", "x(1)"});

// quake.ry(θ) target
// ─────────────────────────────────
// quake.phased_rx(θ, π/2) target
struct RyToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
struct RyToPhasedRx
    : public cudaq::DecompositionPattern<RyToPhasedRxType, cudaq::quake::RyOp> {
  using cudaq::DecompositionPattern<RyToPhasedRxType,
                                    cudaq::quake::RyOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);

    SmallVector<Value> parameters = {angle, pi_2};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RyToPhasedRx, {"ry", "phased_rx"});

// quake.ry(θ) target
// ───────────────────────────────
// quake.s target      (S.S.S = S^dagger)
// quake.s target
// quake.s target
// quake.h target
// quake.rz(θ) target
// quake.h target
// quake.s target
//
// Exact identity Ry(θ) = S . H . Rz(θ) . H . S^dagger. Emitted in circuit
// order with S^dagger expanded as S.S.S (S^4 = I) so the output stays in the
// Rz+Clifford alphabet {H, S, Rz} with no adjoint gates. It is used ahead of
// clifford-t-synthesis so that synthesis only has to handle Rz.
struct RyToRzType; // forward declare the pattern type, defined in the macro
                   // below
struct RyToRz
    : public cudaq::DecompositionPattern<RyToRzType, cudaq::quake::RyOp> {
  using cudaq::DecompositionPattern<RyToRzType,
                                    cudaq::quake::RyOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);

    SmallVector<Value> noControls;
    SmallVector<Value> rzParams = {angle};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::SOp>(loc, target);
    qRewriter.create<cudaq::quake::SOp>(loc, target);
    qRewriter.create<cudaq::quake::SOp>(loc, target);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, rzParams, noControls, target);
    qRewriter.create<cudaq::quake::HOp>(loc, target);
    qRewriter.create<cudaq::quake::SOp>(loc, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RyToRz, {"ry", "s", "h", "rz"});

// quake.ry<adj> (θ) target
// ─────────────────────────────────
// quake.ry(-θ) target
struct RyAdjToRyType; // forward declare the pattern type, defined in the macro
                      // below
struct RyAdjToRy
    : public cudaq::DecompositionPattern<RyAdjToRyType, cudaq::quake::RyOp> {
  using cudaq::DecompositionPattern<RyAdjToRyType,
                                    cudaq::quake::RyOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RyOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RyAdjToRy, {"ry<adj>", "ry"});

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
struct CRzToCX
    : public cudaq::DecompositionPattern<CRzToCXType, cudaq::quake::RzOp> {
  using cudaq::DecompositionPattern<CRzToCXType,
                                    cudaq::quake::RzOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
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
      angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = arith::NegFOp::create(rewriter, loc, halfAngle);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RzOp>(loc, halfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, /*isAdj*/ negControl,
                                         negHalfAngle, noControls, target);
    qRewriter.create<cudaq::quake::XOp>(loc, control, target);

    qRewriter.selectWiresAndReplaceUses(op, ValueRange{control, target});
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(CRzToCX, {"rz(1)", "rz", "x(1)"});

// quake.rz(θ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-θ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct RzToPhasedRxType; // forward declare the pattern type, defined in the
                         // macro below
struct RzToPhasedRx
    : public cudaq::DecompositionPattern<RzToPhasedRxType, cudaq::quake::RzOp> {
  using cudaq::DecompositionPattern<RzToPhasedRxType,
                                    cudaq::quake::RzOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = arith::NegFOp::create(rewriter, loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);
    Value negAngle = arith::NegFOp::create(rewriter, loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    qRewriter.create<cudaq::quake::PhasedRxOp>(loc, parameters, noControls,
                                               target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RzToPhasedRx, {"rz", "phased_rx"});

// quake.rz<adj> (θ) target
// ─────────────────────────────────
// quake.rz(-θ) target
struct RzAdjToRzType; // forward declare the pattern type, defined in the macro
                      // below
struct RzAdjToRz
    : public cudaq::DecompositionPattern<RzAdjToRzType, cudaq::quake::RzOp> {
  using cudaq::DecompositionPattern<RzAdjToRzType,
                                    cudaq::quake::RzOp>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    if (!op.isAdj())
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    angle = arith::NegFOp::create(rewriter, loc, angle);

    // Necessary/Helpful constants
    SmallVector<Value> noControls;
    SmallVector<Value> parameters = {angle};

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RzOp>(loc, parameters, noControls, target);

    qRewriter.selectWiresAndReplaceUses(op, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(RzAdjToRz, {"rz<adj>", "rz"});

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
struct U3ToRotations : public cudaq::DecompositionPattern<U3ToRotationsType,
                                                          cudaq::quake::U3Op> {
  using cudaq::DecompositionPattern<U3ToRotationsType,
                                    cudaq::quake::U3Op>::DecompositionPattern;

  LogicalResult matchAndRewrite(cudaq::quake::U3Op op,
                                PatternRewriter &rewriter) const override {
    if (!isEnabled(cudaq::getKnownNumControls(op)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    SmallVector<Value> controls(op.getControls());
    Value theta = op.getParameters()[0];
    Value phi = op.getParameters()[1];
    Value lam = op.getParameters()[2];

    if (op.isAdj()) {
      theta = arith::NegFOp::create(rewriter, loc, theta);
      // swap the 2nd and 3rd parameter for correctness
      std::swap(phi, lam);
      phi = arith::NegFOp::create(rewriter, loc, phi);
      lam = arith::NegFOp::create(rewriter, loc, lam);
    }

    // Necessary/Helpful constants
    Type angleType = op.getParameter().getType();
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = arith::NegFOp::create(rewriter, loc, pi_2);

    QuakeOperatorCreator qRewriter(rewriter);
    qRewriter.create<cudaq::quake::RzOp>(loc, lam, controls, target);
    qRewriter.create<cudaq::quake::RxOp>(loc, pi_2, controls, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, theta, controls, target);
    qRewriter.create<cudaq::quake::RxOp>(loc, negPi_2, controls, target);
    qRewriter.create<cudaq::quake::RzOp>(loc, phi, controls, target);

    qRewriter.selectWiresAndReplaceUses(op, controls, target);
    rewriter.eraseOp(op);
    return success();
  }
};
REGISTER_DECOMPOSITION_PATTERN(U3ToRotations, {"u3(n)", "rz(n)", "rx(n)"});

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
             cudaq::DecompositionPatternTypeRegistry::entries()) {
          map[patternType.getName().str()] = patternType.instantiate();
        }
        return map;
      }();

  for (auto it = patternTypes.begin(), ie = patternTypes.end(); it != ie;
       ++it) {
    patterns.add(it->second->create(patterns.getContext()));
  }
}
