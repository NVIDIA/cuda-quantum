/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "qir-remove-measurements"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYNVQIRCALLOPS
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
/// Verify that the QIR doesn't have any "bonus" calls to arbitrary code that is
/// not possibly defined in the QIR standard or NVQIR runtime.
struct VerifyNVQIRCallOpsPass
    : public cudaq::opt::impl::VerifyNVQIRCallOpsBase<VerifyNVQIRCallOpsPass> {
  explicit VerifyNVQIRCallOpsPass(const std::vector<llvm::StringRef> &af)
      : VerifyNVQIRCallOpsBase(), allowedFuncs(af) {}

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    bool passFailed = false;
    // Check that a function name is either QIR or NVQIR registered.
    const auto isKnownFunctionName = [&](llvm::StringRef functionName) -> bool {
      if (functionName.startswith("__quantum_"))
        return true;
      static const std::vector<llvm::StringRef> NVQIR_FUNCS = {
          cudaq::opt::NVQIRInvokeWithControlBits,
          cudaq::opt::NVQIRInvokeRotationWithControlBits,
          cudaq::opt::NVQIRInvokeWithControlRegisterOrBits,
          cudaq::opt::NVQIRPackSingleQubitInArray,
          cudaq::opt::NVQIRReleasePackedQubitArray};
      // It must be either NVQIR extension functions or in the allowed list.
      return std::find(NVQIR_FUNCS.begin(), NVQIR_FUNCS.end(), functionName) !=
                 NVQIR_FUNCS.end() ||
             std::find(allowedFuncs.begin(), allowedFuncs.end(),
                       functionName) != allowedFuncs.end();
    };

    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        if (auto calleeAttr = call.getCalleeAttr()) {
          auto funcName = calleeAttr.getValue();
          if (!isKnownFunctionName(funcName)) {
            call.emitOpError("unexpected function call in NVQIR: " + funcName);
            passFailed = true;
            return WalkResult::interrupt();
          }
        } else {
          call.emitOpError("unexpected indirect call in NVQIR");
          passFailed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      } else if (isa<LLVM::InlineAsmOp, LLVM::InvokeOp, LLVM::ResumeOp>(op)) {
        op->emitOpError("unexpected op in NVQIR");
        passFailed = true;
        return WalkResult::interrupt();
      } else if (!isa<LLVM::AddressOfOp, LLVM::AllocaOp, LLVM::BitcastOp,
                      LLVM::ExtractValueOp, LLVM::GEPOp, LLVM::InsertValueOp,
                      LLVM::LoadOp, LLVM::StoreOp>(op)) {
        // No pointers allowed except for the above operations.
        for (auto oper : op->getOperands()) {
          if (isa<LLVM::LLVMPointerType>(oper.getType())) {
            op->emitOpError("unexpected operand in NVQIR");
            passFailed = true;
            return WalkResult::interrupt();
          }
        }
        for (auto oper : op->getResults()) {
          if (isa<LLVM::LLVMPointerType>(oper.getType())) {
            op->emitOpError("unexpected op result in NVQIR");
            passFailed = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (passFailed) {
      emitError(func.getLoc(),
                "function " + func.getName() + " not compatible with NVQIR.");
      signalPassFailure();
    }
  }

private:
  std::vector<llvm::StringRef> allowedFuncs;
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createVerifyNVQIRCallOpsPass(
    const std::vector<llvm::StringRef> &allowedFuncs) {
  return std::make_unique<VerifyNVQIRCallOpsPass>(allowedFuncs);
}
