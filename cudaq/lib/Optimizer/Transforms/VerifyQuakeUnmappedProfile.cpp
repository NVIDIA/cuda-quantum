/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYQUAKEUNMAPPEDPROFILE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

/// Whether \p func is a kernel: an entry point or a function that takes
/// qubits. These are the functions `basis-conversion` converts, so only their
/// gates must be in the basis.
static bool isKernel(func::FuncOp func) {
  return func->hasAttr(cudaq::entryPointAttrName) ||
         llvm::any_of(func.getArgumentTypes(), [](Type type) {
           return isa<cudaq::quake::RefType, cudaq::quake::VeqType>(type);
         });
}

/// With `wires-only`, report why the entry point \p func holds a qubit as
/// a reference. Report each operation that is a known cause: an index that is
/// not a compile-time constant, and `quake.apply_noise`, which takes qubits
/// only as references. Without a known cause, report only the first operation
/// that yields a reference, so that a large register does not repeat the
/// diagnostic for each of its qubits.
static bool verifyWiresOnly(func::FuncOp func) {
  static constexpr llvm::StringLiteral violation =
      "profile violation: quake-unmapped with value semantics requires every "
      "qubit to be a wire";
  bool ok = true;
  const auto report = [&](Operation *op, llvm::StringRef reason) {
    op->emitOpError(violation) << ", but " << reason;
    ok = false;
  };
  if (llvm::any_of(func.getArgumentTypes(),
                   cudaq::quake::isQuantumReferenceType))
    report(func, "the entry point takes a reference");
  Operation *firstReference = nullptr;
  func.walk([&](Operation *op) {
    auto extract = dyn_cast<cudaq::quake::ExtractRefOp>(op);
    if (isa<cudaq::quake::ApplyNoiseOp>(op))
      report(op, "noise applies only to references");
    else if (extract && !extract.hasConstantIndex())
      report(op, "the index is not a compile-time constant, because it is "
                 "known only at run time or comes from a loop that was not "
                 "unrolled");
    else if (!firstReference &&
             llvm::any_of(op->getResultTypes(),
                          cudaq::quake::isQuantumReferenceType))
      firstReference = op;
  });
  if (ok && firstReference)
    report(firstReference, "this operation yields a reference");
  return ok;
}

namespace {

struct VerifyQuakeUnmappedProfile
    : public cudaq::opt::impl::VerifyQuakeUnmappedProfileBase<
          VerifyQuakeUnmappedProfile> {
  using VerifyQuakeUnmappedProfileBase::VerifyQuakeUnmappedProfileBase;

  void runOnOperation() override {
    auto module = getOperation();
    bool ok = true;
    module.walk([&](cudaq::quake::BorrowWireOp borrow) {
      borrow.emitOpError("profile violation: quake-unmapped forbids qubits "
                         "borrowed from a wire set");
      ok = false;
    });

    if (!basis.empty()) {
      auto basisTarget = cudaq::createBasisTarget(*module.getContext(), basis);
      for (auto func : module.getOps<func::FuncOp>()) {
        if (!isKernel(func))
          continue;
        func.walk([&](Operation *op) {
          if (isa_and_nonnull<cudaq::quake::QuakeDialect>(op->getDialect()) &&
              !basisTarget->isLegal(op)) {
            op->emitOpError(
                "profile violation: operation is not in the target basis");
            ok = false;
          }
        });
      }
    }
    if (wiresOnly)
      for (auto func : module.getOps<func::FuncOp>())
        if (func->hasAttr(cudaq::entryPointAttrName))
          ok &= verifyWiresOnly(func);
    if (!ok)
      signalPassFailure();
  }
};

} // namespace
