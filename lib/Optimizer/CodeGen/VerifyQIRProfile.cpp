/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "verify-qir-profile"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYQIRPROFILE
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// @brief Return true if the CallOp is on a FuncOp declaration that
/// is annotated with the cudaq-fnid attribute.
bool isDeviceCallFuncOp(LLVM::CallOp call) {
  // Need the ModuleOp
  auto mod = call->getParentOfType<ModuleOp>();
  // Need the function name
  auto funcNameAttr = call.getCalleeAttr();
  if (!funcNameAttr)
    return false;
  auto funcName = funcNameAttr.getValue();
  // Get the FuncOp declaration
  auto calleeFunc = mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!calleeFunc)
    return false;

  // Should have a passthrough attribute
  auto passthroughAttr = calleeFunc->getAttrOfType<ArrayAttr>("passthrough");
  if (!passthroughAttr)
    return false;

  // Look for ArrayAttr like ["cudaq-fnid", "1565822655"]
  for (auto attr : passthroughAttr) {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr)
      continue;
    if (arrayAttr.size() != 2)
      continue;
    // Get the key StringAttr
    auto key = dyn_cast<StringAttr>(arrayAttr[0]);
    if (!key)
      continue;
    // If cudaq-fnid, we found it
    if (key.getValue() == "cudaq-fnid")
      return true;
  }

  // Not a device call function declaration
  return false;
}
/// Verify that the specific profile QIR code is sane. For now, this simply
/// checks that the QIR doesn't have any "bonus" calls to arbitrary code that is
/// not possibly defined in the QIR standard.
struct VerifyQIRProfilePass
    : public cudaq::opt::impl::VerifyQIRProfileBase<VerifyQIRProfilePass> {
  using VerifyQIRProfileBase::VerifyQIRProfileBase;

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    bool passFailed = false;
    if (!func->hasAttr(cudaq::entryPointAttrName))
      return;
    auto *ctx = &getContext();
    const bool isBaseProfile = convertTo.getValue() == "qir-base";
    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        // Always accept device_call functions
        if (isDeviceCallFuncOp(call))
          return WalkResult::advance();

        auto funcNameAttr = call.getCalleeAttr();
        if (!funcNameAttr)
          return WalkResult::advance();
        auto funcName = funcNameAttr.getValue();
        if (isBaseProfile && (!funcName.startswith("__quantum_") ||
                              funcName.equals(cudaq::opt::QIRCustomOp))) {
          call.emitOpError("unexpected call in QIR base profile");
          passFailed = true;
          return WalkResult::advance();
        }

        // Check that qubits are unique values.
        const std::size_t numOpnds = call.getNumOperands();
        auto qubitTy = cudaq::opt::getQubitType(ctx);
        if (numOpnds > 0)
          for (std::size_t i = 0; i < numOpnds - 1; ++i)
            if (call.getOperand(i).getType() == qubitTy)
              for (std::size_t j = i + 1; j < numOpnds; ++j)
                if (call.getOperand(j).getType() == qubitTy) {
                  auto i1 =
                      call.getOperand(i).getDefiningOp<LLVM::IntToPtrOp>();
                  auto j1 =
                      call.getOperand(j).getDefiningOp<LLVM::IntToPtrOp>();
                  if (i1 && j1 && i1.getOperand() == j1.getOperand()) {
                    call.emitOpError("uses same qubit as multiple operands");
                    passFailed = true;
                    return WalkResult::interrupt();
                  }
                }
        return WalkResult::advance();
      }
      if (isBaseProfile && isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ResumeOp,
                               LLVM::UnreachableOp, LLVM::SwitchOp>(op)) {
        op->emitOpError("QIR base profile does not support control-flow");
        passFailed = true;
      }
      return WalkResult::advance();
    });
    if (passFailed) {
      emitError(func.getLoc(),
                "function " + func.getName() +
                    " not compatible with the QIR base profile.");
      signalPassFailure();
    }
  }
};
} // namespace
