/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYPROFILEPHYSICALQUAKE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Read-only verifier for physical-quake profile stage boundaries.
struct VerifyProfilePhysicalQuake
    : public cudaq::opt::impl::VerifyProfilePhysicalQuakeBase<
          VerifyProfilePhysicalQuake> {
  using VerifyProfilePhysicalQuakeBase::VerifyProfilePhysicalQuakeBase;

  void runOnOperation() override;
};

} // namespace

/// Whether the operation uses a value-form (SSA) quantum type.
static bool hasNonReferenceValue(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), cudaq::quake::isQuantumValueType) ||
      llvm::any_of(op->getResultTypes(), cudaq::quake::isQuantumValueType))
    return true;
  return llvm::any_of(op->getRegions(), [&](Region &region) {
    return llvm::any_of(region.getBlocks(), [&](Block &block) {
      return llvm::any_of(block.getArgumentTypes(),
                          cudaq::quake::isQuantumValueType);
    });
  });
}

/// Whether the operation takes or returns quantum references.
static bool hasQuantumRefValue(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      cudaq::quake::isQuantumReferenceType) ||
         llvm::any_of(op->getResultTypes(),
                      cudaq::quake::isQuantumReferenceType);
}

/// Whether the operation passes an opaque C++/kernel callable,
/// which may capture or allocate quantum state in an unavailable body.
static bool hasCallableValue(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), [](Type t) {
    return isa<cudaq::cc::CallableType, cudaq::cc::IndirectCallableType>(t);
  });
}

/// Whether an operand is a quantum reference.
static bool isQuantumRef(Value v) {
  return cudaq::quake::isQuantumReferenceType(v.getType());
}

static bool hasQuantumRefOperand(Operation *op) {
  return llvm::any_of(op->getOperands(), isQuantumRef);
}

static bool producesQuantumRef(Operation *op) {
  return llvm::any_of(op->getResultTypes(), [](Type t) {
    return cudaq::quake::isQuantumReferenceType(t);
  });
}

static bool hasOpaqueSignature(Type t) {
  return cudaq::quake::isQuantumReferenceType(t) ||
         isa<cudaq::cc::CallableType, cudaq::cc::IndirectCallableType>(t);
}

/// The trusted-classical convention shared with the walk above: an unmarked
/// external declaration with no quantum references or callables in its
/// signature is classical by convention.
static bool trustedClassical(func::FuncOp callee) {
  if (callee->hasAttr(cudaq::kernelAttrName))
    return false;
  auto sig = callee.getFunctionType();
  return llvm::none_of(sig.getInputs(), hasOpaqueSignature) &&
         llvm::none_of(sig.getResults(), hasOpaqueSignature);
}

static bool mayAliasMeasured(Operation *op) {
  op->emitOpError("cannot verify reset-before-reuse: reference may alias a "
                  "measured qubit");
  return false;
}

/// Checks the reset-before-reuse constraint on one function. Obligations
/// arise only from measurements; functions that never touch quantum
/// references or opaque callables need no checking. The scan uses the
/// entry block's operation order, which is program order only for a
/// single-block body without nested regions; anything else is
/// conservatively unverifiable.
static bool checkResetBeforeReuse(func::FuncOp func) {
  Operation *firstMeasure = nullptr;
  bool touchesQuantum = false;
  func.walk([&](Operation *op) {
    if (isa<cudaq::quake::MeasurementInterface>(op)) {
      touchesQuantum = true;
      firstMeasure = firstMeasure ? firstMeasure : op;
    } else if (isa<CallOpInterface>(op) || hasQuantumRefOperand(op) ||
               hasCallableValue(op) || producesQuantumRef(op)) {
      touchesQuantum = true;
    }
  });
  if (!touchesQuantum)
    return true;

  auto &body = func.getBody();
  if (body.empty())
    return true;
  bool straightLine = body.getBlocks().size() == 1 &&
                      !llvm::any_of(body.front(), [](Operation &op) {
                        return op.getNumRegions() != 0;
                      });
  if (!straightLine) {
    Operation *at = firstMeasure ? firstMeasure : func.getOperation();
    at->emitOpError(
        "cannot verify reset-before-reuse: unsupported control flow");
    return false;
  }

  // References with an outstanding reset obligation and the measurements
  // that created them.
  DenseMap<Value, Operation *> obligated;

  for (Operation &opRef : body.front()) {
    Operation *op = &opRef;

    // Measuring again is exempt and keeps the obligation.
    if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
      for (Value target : meas.getTargets())
        obligated.try_emplace(target, op);
      continue;
    }

    // A reset or deallocation ends the obligation of the exact reference.
    // It cannot be proven to apply to a different reference that may alias
    // it.
    if (isa<cudaq::quake::ResetOp>(op) || isa<cudaq::quake::DeallocOp>(op)) {
      for (Value target : op->getOperands())
        obligated.erase(target);
      if (!obligated.empty())
        return mayAliasMeasured(op);
      continue;
    }

    // Bookkeeping that is not quantum reuse.
    if (auto evince = dyn_cast<cudaq::quake::EvinceOp>(op)) {
      if (evince.getCompilerGenerated())
        continue;
    } else if (isa<cudaq::quake::ExtractRefOp, cudaq::quake::ConcatOp,
                   cudaq::quake::AllocaOp>(op)) {
      continue;
    }

    // Gates: targets and controls alike are uses of the measured qubit.
    if (isa<cudaq::quake::OperatorInterface>(op)) {
      auto operands = op->getOperands();
      auto exact = llvm::find_if(operands, [&](Value used) {
        return isQuantumRef(used) && obligated.count(used);
      });
      if (exact != operands.end()) {
        op->emitOpError("profile violation: measured qubit is reused "
                        "without a required reset")
                .attachNote(obligated.find(*exact)->second->getLoc())
            << "measured here";
        return false;
      }
      if (!obligated.empty() && hasQuantumRefOperand(op))
        return mayAliasMeasured(op);
      continue;
    }

    // Calls into an available in-module body with no quantum reference or
    // callable crossing the boundary and no quantum reference produced are
    // checked in the callee itself; an unavailable callee is trusted only
    // under the shared trusted-classical convention, which never extends to
    // quantum invocations (any call form in the quake dialect, e.g. apply
    // or the retained call_by_ref). Everything else is conservatively
    // unverifiable.
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (!hasQuantumRefOperand(op) && !hasCallableValue(op) &&
          !producesQuantumRef(op)) {
        auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
        auto callee =
            symbol
                ? SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, symbol)
                : nullptr;
        if (callee && !callee.isExternal())
          continue;
        if (callee && !isa<cudaq::quake::QuakeDialect>(op->getDialect()) &&
            trustedClassical(callee))
          continue;
      }
      op->emitOpError("cannot verify reset-before-reuse: operation cannot "
                      "be proven not to reuse a measured qubit");
      return false;
    }

    // Unclassified operations (applications, escapes through returns or
    // captures, quantum-reference producers) cannot be proven not to
    // measure or reuse a qubit.
    if (hasQuantumRefOperand(op) || hasCallableValue(op) ||
        producesQuantumRef(op)) {
      op->emitOpError("cannot verify reset-before-reuse: operation cannot "
                      "be proven not to reuse a measured qubit");
      return false;
    }
  }
  return true;
}

/// Verifies the reset-before-reuse constraint on every function: a qubit
/// measured in a function must be reset before any subsequent
/// non-measurement quantum use, controls included. Reference bookkeeping,
/// deallocation, and repeated measurement are exempt. A proven missing
/// reset fails with `profile violation`; unprovable cases fail with
/// `cannot verify`.
static LogicalResult verifyResetBeforeReuse(ModuleOp module) {
  bool ok = true;
  module->walk([&](func::FuncOp func) {
    if (!checkResetBeforeReuse(func))
      ok = false;
  });
  return success(ok);
}

void VerifyProfilePhysicalQuake::runOnOperation() {
  auto module = getOperation();

  if (measurements != "any" && measurements != "expanded") {
    module.emitError(
        "invalid value '" + measurements +
        "' for option 'measurements' (expected 'any' or 'expanded')");
    signalPassFailure();
    return;
  }
  const bool expandedForm = measurements == "expanded";

  // An empty basis imposes no gate-basis restriction.
  std::unique_ptr<ConversionTarget> basisTarget;
  if (!basis.empty())
    basisTarget = cudaq::createBasisTarget(*module.getContext(), basis);

  // Resolves from the call's nearest symbol table; failed, non-function,
  // or bodyless resolution is unprovable (never a crash).
  auto hasBody = [](Operation *from, SymbolRefAttr symbol) {
    auto func =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(from, symbol);
    return func && !func.isExternal();
  };

  bool failed = false;
  module.walk([&](Operation *op) {
    if (hasNonReferenceValue(op)) {
      op->emitOpError("profile violation: physical-quake requires "
                      "reference-form Quake (no !quake.wire, "
                      "!quake.control, or !quake.cable values)");
      failed = true;
    }
    if (expandedForm) {
      if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
        auto targets = meas.getTargets();
        if (targets.size() != 1) {
          op->emitOpError("profile violation: expanded measurement form "
                          "requires exactly one !quake.ref target (got ")
              << targets.size() << " targets)";
          failed = true;
        } else if (!isa<cudaq::quake::RefType>(targets[0].getType())) {
          op->emitOpError("profile violation: expanded measurement form "
                          "requires a !quake.ref target (got ")
              << targets[0].getType() << ")";
          failed = true;
        }
      }
    }

    if (!basisTarget && !expandedForm)
      return;

    // Applications and calls whose gates cannot be accounted for cannot be
    // proven against the basis: reject as unverifiable, never approve.
    if (auto apply = dyn_cast<cudaq::quake::ApplyOp>(op)) {
      if (basisTarget && !apply.getControls().empty()) {
        op->emitOpError("cannot verify: controlled kernel application is "
                        "not provable against the target basis");
        failed = true;
      } else if (apply.getIndirectCallee() || !apply.getCallee() ||
                 !hasBody(op, *apply.getCallee())) {
        op->emitOpError("cannot verify: applied kernel body is not available");
        failed = true;
      }
      return;
    }
    // compute_action's callable bodies are opaque to this pass.
    if (isa<cudaq::quake::ComputeActionOp>(op)) {
      op->emitOpError("cannot verify: compute_action bodies are not available");
      failed = true;
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
      func::FuncOp callee;
      if (symbol)
        callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, symbol);
      // An indirect or unresolved callee's body is statically unavailable,
      // as is a declaration-only one; accept the latter only when provably
      // classical (no quantum-reference or opaque-callable operands and
      // not declared a quantum kernel).
      bool unavailable = !callee || callee.isExternal();
      bool provablyClassical = callee && !hasQuantumRefValue(op) &&
                               !hasCallableValue(op) &&
                               !callee->hasAttr(cudaq::kernelAttrName);
      if (unavailable && !provablyClassical) {
        op->emitOpError("cannot verify: called function body is not available");
        failed = true;
      }
      return;
    }

    if (!basisTarget)
      return;

    if (!op->getDialect() ||
        !isa<cudaq::quake::QuakeDialect>(op->getDialect()) ||
        basisTarget->isLegal(op))
      return;

    // A gate with a dynamically-sized control vector cannot be proven
    // against finite-arity entries: reject as unverifiable, distinct
    // from a proven violation (name absent from the basis).
    if (auto optor = dyn_cast<cudaq::quake::OperatorInterface>(op);
        optor && !cudaq::getKnownNumControls(optor)) {
      auto name = optor->getName().stripDialect();
      if (llvm::any_of(basis, [&](const std::string &b) {
            return StringRef(b).take_front(b.find_first_of("(<")) == name;
          })) {
        op->emitOpError("cannot verify: number of controls is not "
                        "statically known and the basis has no "
                        "unbounded-arity entry for it");
        failed = true;
        return;
      }
    }
    op->emitOpError("profile violation: operation is not in the target basis");
    failed = true;
  });

  // Reset-before-reuse checking is independent of the walk above.
  if (resetBeforeReuse && !succeeded(verifyResetBeforeReuse(module)))
    failed = true;

  if (failed)
    signalPassFailure();
}
