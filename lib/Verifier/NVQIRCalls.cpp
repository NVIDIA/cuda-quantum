/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Verifier/NVQIRCalls.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

constexpr const char *qirSpecPrefixes[] = {"__quantum_"};

constexpr const char *llvmIntrinsicPrefixes[] = {
    "llvm.memcpy.", "llvm.memmove.", "llvm.memset."};

constexpr const char *nvqirFuncs[] = {
    cudaq::opt::NVQIRInvokeWithControlBits,           // obsolete
    cudaq::opt::NVQIRInvokeRotationWithControlBits,   // obsolete
    cudaq::opt::NVQIRInvokeWithControlRegisterOrBits, // obsolete
    cudaq::opt::NVQIRGeneralizedInvokeAny,
    cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex32,
    cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex64,
    cudaq::getNumQubitsFromCudaqState,
    cudaq::createCudaqStateFromDataComplexF32,
    cudaq::createCudaqStateFromDataComplexF64,
    cudaq::createCudaqStateFromDataF32,
    cudaq::createCudaqStateFromDataF64,
    cudaq::deleteCudaqState};

constexpr const char *libcFuncs[] = {"malloc", "free", "memcpy", "memset"};

// Helper function to verify that \p name is a valid NVQIR function and can be
// called/referenced.
static bool isVerifiedFunction(StringRef name,
                               const SmallVector<StringRef> &goldenFuncs) {
  auto prefixCheck = [&](const char *prefix) {
    return name.startswith(prefix);
  };

  // Check if name has an accepted QIR or LLVM intrinsic prefix.
  if (std::find_if(std::begin(qirSpecPrefixes), std::end(qirSpecPrefixes),
                   prefixCheck) != std::end(qirSpecPrefixes) ||
      std::find_if(std::begin(llvmIntrinsicPrefixes),
                   std::end(llvmIntrinsicPrefixes),
                   prefixCheck) != std::end(llvmIntrinsicPrefixes))
    return true;

  // Check against the sets of full names. If the name is not in any of these
  // collections, consider it invalid.
  return std::find(std::begin(nvqirFuncs), std::end(nvqirFuncs), name) !=
             std::end(nvqirFuncs) ||
         std::find(std::begin(libcFuncs), std::end(libcFuncs), name) !=
             std::end(libcFuncs) ||
         std::find(goldenFuncs.begin(), goldenFuncs.end(), name) !=
             goldenFuncs.end();
}

LogicalResult cudaq::verify::checkNvqirCalls(ModuleOp module) {
  // Collect functions that are defined in the module. They are golden.
  SmallVector<StringRef> goldenFuncs;
  for (auto &artifact : module)
    if (auto func = dyn_cast<LLVM::LLVMFuncOp>(artifact))
      if (!func.empty())
        goldenFuncs.push_back(func.getName());

  if (goldenFuncs.empty())
    return success(); // module has nothing of interest.

  auto walkRes = module.walk([&](Operation *op) {
    if (auto call = dyn_cast<LLVM::CallOp>(op)) {
      if (auto calleeAttr = call.getCalleeAttr()) {
        auto funcName = calleeAttr.getValue();
        if (!isVerifiedFunction(funcName, goldenFuncs)) {
          call.emitOpError("unexpected function call in NVQIR: " + funcName);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      call.emitOpError("unexpected indirect call in NVQIR");
      return WalkResult::interrupt();
    }
    if (isa<LLVM::InlineAsmOp, LLVM::InvokeOp, LLVM::ResumeOp>(op)) {
      op->emitOpError("unexpected op in NVQIR");
      return WalkResult::interrupt();
    }
    if (!isa<LLVM::AddressOfOp, LLVM::AllocaOp, LLVM::BitcastOp,
             LLVM::ExtractValueOp, LLVM::GEPOp, LLVM::InsertValueOp,
             LLVM::IntToPtrOp, LLVM::SelectOp, LLVM::LoadOp, LLVM::StoreOp>(
            op)) {
      // No pointers allowed except for the above operations.
      for (auto oper : op->getOperands())
        if (isa<LLVM::LLVMPointerType>(oper.getType())) {
          op->emitOpError("unexpected memory operand in NVQIR");
          return WalkResult::interrupt();
        }
      for (auto oper : op->getResults())
        if (isa<LLVM::LLVMPointerType>(oper.getType())) {
          op->emitOpError("unexpected memory result in NVQIR");
          return WalkResult::interrupt();
        }
    }
    return WalkResult::advance();
  });

  if (!walkRes.wasInterrupted())
    return success();

  emitError(module.getLoc(), "module has calls not compatible with NVQIR.");
  return failure();
}
