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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYPROFILEPHYSICALQUAKE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Read-only verifier pass for the physical-quake profile.
struct VerifyProfilePhysicalQuake
    : public cudaq::opt::impl::VerifyProfilePhysicalQuakeBase<
          VerifyProfilePhysicalQuake> {
  using VerifyProfilePhysicalQuakeBase::VerifyProfilePhysicalQuakeBase;

  void runOnOperation() override;
};

} // namespace

/// Whether the operation passes an opaque C++/kernel callable, whose body is
/// statically unavailable.
static bool hasCallableOperand(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), [](Type t) {
    return isa<cudaq::cc::CallableType, cudaq::cc::IndirectCallableType>(t);
  });
}

/// Resolve the callee of \p call, or nullptr for indirect or unresolved
/// callees.
static func::FuncOp lookupCallee(CallOpInterface call) {
  auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  if (!symbol)
    return nullptr;
  return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, symbol);
}

/// Whether we are able to recurse verification into the callee of \p call.
///
/// A call is supported if:
/// - the body of the callee is available (and hence can be verified)
/// - the callee is external but deemed purely classical based on its signature
static bool isSupportedCall(CallOpInterface call) {
  auto isQuantumOrCallableType = [](Type t) {
    return cudaq::quake::isQuantumReferenceType(t) ||
           isa<cudaq::cc::CallableType, cudaq::cc::IndirectCallableType>(t);
  };

  auto callee = lookupCallee(call);
  if (!callee)
    return false;
  if (!callee.isExternal())
    return true;
  if (isa<cudaq::quake::QuakeDialect>(call->getDialect()) ||
      callee->hasAttr(cudaq::kernelAttrName))
    return false;
  auto sig = callee.getFunctionType();
  return llvm::none_of(sig.getInputs(), isQuantumOrCallableType) &&
         llvm::none_of(sig.getResults(), isQuantumOrCallableType);
}

/// Physical-quake requires reference-form Quake: no wire/control/cable
/// values anywhere, including in function signatures.
static LogicalResult checkReferenceForm(Operation *op) {
  bool valueForm =
      llvm::any_of(op->getOperandTypes(), cudaq::quake::isQuantumValueType);
  valueForm |=
      llvm::any_of(op->getResultTypes(), cudaq::quake::isQuantumValueType);
  if (auto func = dyn_cast<FunctionOpInterface>(op)) {
    valueForm |=
        llvm::any_of(func.getArgumentTypes(), cudaq::quake::isQuantumValueType);
    valueForm |=
        llvm::any_of(func.getResultTypes(), cudaq::quake::isQuantumValueType);
  }
  if (!valueForm)
    return success();
  return op->emitOpError("profile violation: physical-quake requires "
                         "reference-form Quake (no !quake.wire, "
                         "!quake.control, or !quake.cable values)");
}

/// With `measurement-type=single-qubit`, every measurement operates on
/// exactly one !quake.ref.
static LogicalResult checkMeasurementForm(Operation *op) {
  auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op);
  if (!meas)
    return success();
  auto targets = meas.getTargets();
  if (targets.size() != 1)
    return op->emitOpError("profile violation: single-qubit measurement form "
                           "requires exactly one !quake.ref target (got ")
           << targets.size() << " targets)";
  if (!isa<cudaq::quake::RefType>(targets[0].getType()))
    return op->emitOpError("profile violation: single-qubit measurement form "
                           "requires a !quake.ref target (got ")
           << targets[0].getType() << ")";
  return success();
}

/// Calls whose bodies cannot be accounted for cannot be proven against the
/// profile: reject as unverifiable.
static LogicalResult checkInvocation(Operation *op, bool basisRestricted) {
  // A callable passed as argument has an opaque body.
  if (hasCallableOperand(op))
    return op->emitOpError("cannot verify: callable body is not available");
  // A controlled application composes arities through the callee, which the
  // basis cannot account for.
  if (basisRestricted)
    if (auto apply = dyn_cast<cudaq::quake::ApplyOp>(op);
        apply && !apply.getControls().empty())
      return op->emitOpError(
          "cannot verify: inferring target basis for controlled kernel "
          "application is currently not supported");
  if (auto call = dyn_cast<CallOpInterface>(op); call && !isSupportedCall(call))
    return op->emitOpError("cannot verify: called function body is not "
                           "available");
  return success();
}

/// Every quake operation must be a member of the requested target basis.
static LogicalResult checkBasis(Operation *op, ConversionTarget &target) {
  if (!op->getDialect() || !isa<cudaq::quake::QuakeDialect>(op->getDialect()) ||
      target.isLegal(op))
    return success();
  return op->emitOpError(
      "profile violation: operation is not in the target basis");
}

void VerifyProfilePhysicalQuake::runOnOperation() {
  auto module = getOperation();

  if (measurementType != "any" && measurementType != "single-qubit") {
    module.emitError(
        "invalid value '" + measurementType +
        "' for option 'measurement-type' (expected 'any' or 'single-qubit')");
    signalPassFailure();
    return;
  }
  const bool singleQubitForm = measurementType == "single-qubit";

  if (resetAfterMeasure)
    module.emitWarning("reset-after-measure is not currently validated by "
                       "verify-profile-physical-quake");

  // An empty basis imposes no gate-basis restriction.
  std::unique_ptr<ConversionTarget> basisTarget;
  if (!basis.empty())
    basisTarget = cudaq::createBasisTarget(*module.getContext(), basis);

  bool ok = true;
  module.walk([&](Operation *op) {
    ok &= succeeded(checkReferenceForm(op));
    if (singleQubitForm)
      ok &= succeeded(checkMeasurementForm(op));
    if (basisTarget || singleQubitForm)
      ok &= succeeded(checkInvocation(op, /*basisRestricted=*/!basis.empty()));
    if (basisTarget)
      ok &= succeeded(checkBasis(op, *basisTarget));
  });

  if (!ok)
    signalPassFailure();
}
