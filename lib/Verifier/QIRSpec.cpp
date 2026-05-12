/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Verifier/QIRSpec.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"

/**
   \file

   This file implements checks for the some varied miscellaneous conformance
   restrictions and limitations imposed on LLVM instructions as mandated by the
   QIR Specification. This code is necessarily run on an LLVM Module containing
   LLVM instructions.

   More such checks may be added in the future.

   These checks are considered \e fatal errors. If the generated code was not
   lowered to conform to the QIR Specification, the compilation is considered to
   have failed and corrective action (source changes) will be required.
 */

using namespace mlir;

static bool isValidIntegerArithmeticInstruction(llvm::Instruction &inst) {
  const auto isValidIntegerBinaryInst = [](const auto &inst) {
    if (!llvm::isa<llvm::BinaryOperator>(inst))
      return false;
    const auto opCode = inst.getOpcode();
    static const std::vector<int> integerOps = {
        llvm::BinaryOperator::Add,  llvm::BinaryOperator::Sub,
        llvm::BinaryOperator::Mul,  llvm::BinaryOperator::UDiv,
        llvm::BinaryOperator::SDiv, llvm::BinaryOperator::URem,
        llvm::BinaryOperator::SRem, llvm::BinaryOperator::And,
        llvm::BinaryOperator::Or,   llvm::BinaryOperator::Xor,
        llvm::BinaryOperator::Shl,  llvm::BinaryOperator::LShr,
        llvm::BinaryOperator::AShr};
    return std::find(integerOps.begin(), integerOps.end(), opCode) !=
           integerOps.end();
  };

  return isValidIntegerBinaryInst(inst) ||
         llvm::isa<llvm::ICmpInst, llvm::ZExtInst, llvm::SExtInst,
                   llvm::TruncInst, llvm::SelectInst, llvm::PHINode>(inst);
}

static bool isValidFloatingArithmeticInstruction(llvm::Instruction &inst) {
  const auto isValidFloatBinaryInst = [](const auto &inst) {
    if (!llvm::isa<llvm::BinaryOperator>(inst))
      return false;
    const auto opCode = inst.getOpcode();
    static const std::vector<int> floatOps = {
        llvm::BinaryOperator::FAdd, llvm::BinaryOperator::FSub,
        llvm::BinaryOperator::FMul, llvm::BinaryOperator::FDiv,
        llvm::Instruction::FRem};
    return std::find(floatOps.begin(), floatOps.end(), opCode) !=
           floatOps.end();
  };

  return isValidFloatBinaryInst(inst) || llvm::isa<llvm::FCmpInst>(inst) ||
         llvm::isa<llvm::FPExtInst>(inst) ||
         llvm::isa<llvm::FPTruncInst>(inst) ||
         llvm::isa<llvm::SelectInst>(inst) || llvm::isa<llvm::PHINode>(inst);
}

static bool isValidOutputCallInstruction(llvm::Instruction &inst) {
  if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
    auto *calledFunc = call->getCalledFunction();
    if (!calledFunc)
      return false;
    std::string name = calledFunc->getName().str();
    std::vector<const char *> outputFunctions{
        cudaq::opt::QIRBoolRecordOutput, cudaq::opt::QIRIntegerRecordOutput,
        cudaq::opt::QIRDoubleRecordOutput, cudaq::opt::QIRTupleRecordOutput,
        cudaq::opt::QIRArrayRecordOutput};
    return std::find(outputFunctions.begin(), outputFunctions.end(), name) ==
           outputFunctions.end();
  }
  return false;
}

LogicalResult cudaq::verifier::verifyLLVMInstructions(
    llvm::Module *llvmModule, cudaq::verifier::LLVMVerifierOptions options) {

  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        bool isValidBaseProfileInstruction =
            llvm::isa<llvm::CallBase>(inst) ||
            llvm::isa<llvm::BranchInst>(inst) ||
            llvm::isa<llvm::ReturnInst>(inst);
        bool isValidAdaptiveProfileInstruction = isValidBaseProfileInstruction;
        if (options.isBaseProfile && !isValidBaseProfileInstruction) {
          llvm::errs() << "QIR verification error - invalid instruction found: "
                       << inst << " (base profile)\n";
          if (!options.allowAllInstructions)
            return failure();
        } else if (options.isAdaptiveProfile &&
                   !isValidAdaptiveProfileInstruction) {
          const bool isValidIntExtension =
              options.integerComputations &&
              isValidIntegerArithmeticInstruction(inst);

          const bool isValidFloatExtension =
              options.floatComputations &&
              isValidFloatingArithmeticInstruction(inst);

          const bool isValidOutputCall = isValidOutputCallInstruction(inst);
          if (!isValidIntExtension && !isValidFloatExtension &&
              !isValidOutputCall) {
            llvm::errs()
                << "QIR verification error - invalid instruction found: "
                << inst << " (adaptive profile)\n";
            if (!options.allowAllInstructions)
              return failure();
          }
        }
        auto call = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);
        if (call)
          for (auto &arg : call->args()) {
            auto constExpr = llvm::dyn_cast_or_null<llvm::ConstantExpr>(arg);
            if (constExpr &&
                constExpr->getOpcode() != llvm::Instruction::GetElementPtr &&
                constExpr->getOpcode() != llvm::Instruction::IntToPtr &&
                constExpr->getOpcode() != llvm::Instruction::BitCast) {
              llvm::errs()
                  << "QIR verification error - invalid instruction found: "
                  << *constExpr << " (call argument)\n";
              if (!options.allowAllInstructions)
                return failure();
            }
          }
      }
  return success();
}
