/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Environment.h"
#include "Logger.h"
#include "Timing.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/OptUtils.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"

namespace cudaq {

bool setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return false;

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine)
    return false;

  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  return true;
}

void optimizeLLVM(llvm::Module *module) {
  auto optPipeline = cudaq::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(module))
    throw std::runtime_error("Failed to optimize LLVM IR ");

  // Remove memory attributes from entry_point functions because the optimizer
  // sometimes applies it to degenerate cases (empty programs), and IonQ cannot
  // support that.
  for (llvm::Function &func : *module)
    if (func.hasFnAttribute(cudaq::opt::QIREntryPointAttrName))
      func.removeFnAttr(llvm::Attribute::Memory);
}

void applyWriteOnlyAttributes(llvm::Module *llvmModule) {
  // Note that we only need to inspect QIRMeasureBody because MeasureCallConv
  // and MeasureToRegisterCallConv have already been called, so only
  // QIRMeasureBody remains.
  const unsigned int arg_num = 1;

  // Apply attribute to measurement function declaration
  if (auto func = llvmModule->getFunction(cudaq::opt::QIRMeasureBody)) {
    func->addParamAttr(arg_num, llvm::Attribute::WriteOnly);
  }

  // Apply to measurement function calls
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);
        if (callInst && callInst->getCalledFunction()) {
          auto calledFunc = callInst->getCalledFunction();
          auto funcName = calledFunc->getName();
          if (funcName == cudaq::opt::QIRMeasureBody)
            callInst->addParamAttr(arg_num, llvm::Attribute::WriteOnly);
        }
      }
}

bool isValidIntegerArithmeticInstruction(llvm::Instruction &inst) {
  // Not a valid adaptive profile instruction
  // Check if it's in the extended instruction set
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

  return isValidIntegerBinaryInst(inst) || llvm::isa<llvm::ICmpInst>(inst) ||
         llvm::isa<llvm::ZExtInst>(inst) || llvm::isa<llvm::SExtInst>(inst) ||
         llvm::isa<llvm::TruncInst>(inst) ||
         llvm::isa<llvm::SelectInst>(inst) || llvm::isa<llvm::PHINode>(inst);
}

bool isValidFloatingArithmeticInstruction(llvm::Instruction &inst) {
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
         llvm::isa<llvm::FPExtInst>(inst) || llvm::isa<llvm::FPTruncInst>(inst);
}

bool isValidOutputCallInstruction(llvm::Instruction &inst) {
  // Not a valid adaptive profile instruction
  // Check if it's an record output call.
  if (auto *call = dyn_cast<llvm::CallBase>(&inst)) {
    auto name = call->getCalledFunction()->getName().str();
    std::vector<const char *> outputFunctions{
        cudaq::opt::QIRBoolRecordOutput, cudaq::opt::QIRIntegerRecordOutput,
        cudaq::opt::QIRDoubleRecordOutput, cudaq::opt::QIRTupleRecordOutput,
        cudaq::opt::QIRArrayRecordOutput};
    return std::find(outputFunctions.begin(), outputFunctions.end(),
                     name.c_str()) == outputFunctions.end();
  }
  return false;
}

/// @brief Add module flags according to the spec:
/// https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Adaptive_Profile.md#module-flags-metadata
void applyQIRAdaptiveCapabilitiesAttributes(llvm::Module *llvmModule) {
  llvm::DenseMap<std::size_t, bool> intPrecisions;
  llvm::DenseMap<std::size_t, bool> floatPrecisions;
  std::size_t retCount = 0;
  bool hasMultipleTargetBranching = false;
  std::uint64_t backwardBranching = 0;
  bool hasIRFunctions = false;

  for (llvm::Function &func : *llvmModule) {
    std::size_t funcRetCount = 0;
    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : block) {
        // Collect information to set `multiple_return_points` module flag.
        if (inst.getOpcode() == llvm::Instruction::Ret)
          funcRetCount++;

        // Collect information to set `multiple_target_branching` module flag.
        if (inst.getOpcode() == llvm::Instruction::Switch)
          hasMultipleTargetBranching = true;

        // Collect information to set `backwards_branching` module flag.
        if (auto *br = dyn_cast<llvm::BranchInst>(&inst)) {
          bool isLoop = false;
          for (auto successor : br->successors()) {
            if (successor == &block)
              isLoop = true;
          }
          if (isLoop) {
            // The `backwardBranching` value is a 2-bit integer where bit 0
            // indicates presence of simple iterations, and bit 1 indicates
            // presence of conditionally terminating loops, i.e. loops with
            // an exit that depends on a measurement.
            auto condition = br->getCondition();
            if (auto *call = dyn_cast<llvm::CallBase>(condition)) {
              if (call->getCalledFunction()->getName().str() ==
                  cudaq::opt::QIRReadResultBody)
                backwardBranching |= (std::uint64_t)2;
            } else
              backwardBranching |= (std::uint64_t)1;
          }
        }

        // Collect information to set `int_computations` and
        // `float_computations` module flags.
        if (isValidIntegerArithmeticInstruction(inst) ||
            isValidFloatingArithmeticInstruction(inst)) {
          for (std::size_t i = 0; i < inst.getNumOperands(); i++) {
            auto ty = inst.getOperand(i)->getType();
            if (ty->isIntegerTy())
              intPrecisions[ty->getScalarSizeInBits()] = true;
            else if (ty->isFloatingPointTy())
              floatPrecisions[ty->getScalarSizeInBits()] = true;
          }
        }

        // Collect information to set `if_functions` module flag.
        if (auto *call = dyn_cast<llvm::CallBase>(&inst)) {
          auto name = call->getCalledFunction()->getName().str();
          if (!name.starts_with("__quantum__"))
            hasIRFunctions = true;
        }
      }
    }
    retCount = std::max(funcRetCount, retCount);
  }

  std::string intPrecisionStr;
  llvm::SmallVector<std::size_t> intPrecisionsVec;
  for (auto &[k, v] : intPrecisions)
    if (v)
      intPrecisionsVec.push_back(k);
  std::sort(intPrecisionsVec.begin(), intPrecisionsVec.end());
  for (auto k : intPrecisionsVec) {
    if (!intPrecisionStr.empty())
      intPrecisionStr += ",";
    intPrecisionStr += "i" + std::to_string(k);
  }

  std::string floatPrecisionStr;
  llvm::SmallVector<std::size_t> floatPrecisionsVec;
  for (auto &[k, v] : floatPrecisions)
    if (v)
      floatPrecisionsVec.push_back(k);
  std::sort(floatPrecisionsVec.begin(), floatPrecisionsVec.end());
  for (auto k : floatPrecisionsVec) {
    if (!floatPrecisionStr.empty())
      floatPrecisionStr += ",";
    floatPrecisionStr += "f" + std::to_string(k);
  }

  auto &llvmContext = llvmModule->getContext();
  auto trueValue =
      llvm::ConstantInt::getTrue(llvm::Type::getInt1Ty(llvmContext));

  if (hasIRFunctions)
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              cudaq::opt::QIRIrFunctionsFlagName, trueValue);

  if (!intPrecisionStr.empty()) {
    llvm::Constant *intPrecisionValue =
        llvm::ConstantDataArray::getString(llvmContext, intPrecisionStr, false);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              cudaq::opt::QIRIntComputationsFlagName,
                              intPrecisionValue);
  }
  if (!floatPrecisionStr.empty()) {
    llvm::Constant *floatPrecisionValue = llvm::ConstantDataArray::getString(
        llvmContext, floatPrecisionStr, false);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              cudaq::opt::QIRFloatComputationsFlagName,
                              floatPrecisionValue);
  }

  auto backwardsBranchingValue = llvm::ConstantInt::getIntegerValue(
      llvm::Type::getIntNTy(llvmContext, 2),
      llvm::APInt(2, backwardBranching, false));
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRBackwardsBranchingFlagName,
                            backwardsBranchingValue);

  if (hasMultipleTargetBranching)
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              cudaq::opt::QIRMultipleTargetBranchingFlagName,
                              trueValue);

  if (retCount > 1)
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              cudaq::opt::QIRMultipleReturnPointsFlagName,
                              trueValue);
}

// Once a call to a function with irreversible attribute is seen, no more calls
// to reversible functions are allowed. This is somewhat of an implied
// specification because the specification describes the program in terms of 4
// sequential blocks. The 2nd block contains reversible operations, and the 3rd
// block contains irreversible operations (measurements), and the blocks may not
// overlap.
// Reference:
// https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md?plain=1#L237
mlir::LogicalResult
verifyBaseProfileMeasurementOrdering(llvm::Module *llvmModule) {
  bool irreversibleSeenYet = false;
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);

        if (callInst && callInst->getCalledFunction()) {
          auto calledFunc = callInst->getCalledFunction();
          auto funcName = calledFunc->getName();
          bool isIrreversible =
              calledFunc->hasFnAttribute(cudaq::opt::QIRIrreversibleFlagName);
          bool isReversible = !isIrreversible;
          bool isOutputFunction = funcName == cudaq::opt::QIRRecordOutput;
          if (isReversible && !isOutputFunction && irreversibleSeenYet) {
            llvm::errs() << "error: reversible function " << funcName
                         << " came after irreversible function\n";
            return mlir::failure();
          }
          if (isIrreversible)
            irreversibleSeenYet = true;
        }
      }
  return mlir::success();
}

// Verify that output recording calls
// 1) Have the nonnull attribute on any i8* parameters
// 2) Have unique names
mlir::LogicalResult verifyOutputCalls(llvm::CallBase *callInst,
                                      std::set<std::string> &outputList) {
  int iArg = 0;
  for (auto &arg : callInst->args()) {
    auto myArg = arg->getType();
    auto ptrTy = dyn_cast_or_null<llvm::PointerType>(myArg);
    // If we're dealing with the i8* parameters
    if (ptrTy != nullptr &&
        ptrTy->getNonOpaquePointerElementType()->isIntegerTy(8)) {
      // Verify that it has the nonnull attribute
      if (!callInst->paramHasAttr(iArg, llvm::Attribute::NonNull)) {
        llvm::errs() << "error - nonnull attribute is missing from i8* "
                        "parameter of "
                     << cudaq::opt::QIRRecordOutput << " function\n";
        return mlir::failure();
      }

      // Lookup the string value from IR that looks like this:
      // clang-format off
      // i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.723000, i64 0, i64 0))
      // clang-format on
      auto constExpr = llvm::dyn_cast_or_null<llvm::ConstantExpr>(arg);
      if (constExpr &&
          constExpr->getOpcode() == llvm::Instruction::GetElementPtr) {
        llvm::Value *globalValue = constExpr->getOperand(0);
        auto globalVar =
            llvm::dyn_cast_or_null<llvm::GlobalVariable>(globalValue);

        // Get the string value of the output name and compare it against
        // the previously identified names in outputList[].
        if (globalVar && globalVar->hasInitializer()) {
          auto constDataArray = llvm::dyn_cast_or_null<llvm::ConstantDataArray>(
              globalVar->getInitializer());
          if (constDataArray) {
            std::string strValue = constDataArray->getAsCString().str();
            if (outputList.find(strValue) != outputList.end()) {
              llvm::errs() << "error - duplicate output name (" << strValue
                           << ") found!\n";
              return mlir::failure();
            }
          }
        }
      }
    }

    iArg++;
  }
  return mlir::success();
}

// Loop through the arguments in a call and verify that they are all constants
mlir::LogicalResult verifyConstArguments(llvm::CallBase *callInst) {
  int iArg = 0;
  auto func = callInst ? callInst->getCalledFunction() : nullptr;
  auto funcName = func ? func->getName() : "N/A";
  for (auto &arg : callInst->args()) {
    // Try casting to Constant Type. Fail if it's not a constant.
    if (!dyn_cast_or_null<llvm::Constant>(arg)) {
      llvm::errs() << "error: argument #" << iArg << " ('" << *arg
                   << "') in call " << funcName << " is not a constant\n";
      return mlir::failure();
    }
    iArg++;
  }
  return mlir::success();
}

// Loop over the recording output functions and verify their characteristics
mlir::LogicalResult verifyOutputRecordingFunctions(llvm::Module *llvmModule,
                                                   bool isBaseProfile) {
  for (llvm::Function &func : *llvmModule) {
    std::set<std::string> outputList;
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);
        auto func = callInst ? callInst->getCalledFunction() : nullptr;
        // All call arguments must be constants if this is a base profile
        if (isBaseProfile && func && failed(verifyConstArguments(callInst)))
          return mlir::failure();
        // If it's an output function, do additional verification
        if (func && func->getName() == cudaq::opt::QIRRecordOutput)
          if (failed(verifyOutputCalls(callInst, outputList)))
            return mlir::failure();
      }
  }
  return mlir::success();
}

// Convert a `nullptr` or `inttoptr (i64 1 to Ptr)` into an integer
std::size_t getArgAsInteger(llvm::Value *arg) {
  std::size_t ret = 0; // handles the nullptr case
  // Now handle the `inttoptr (i64 1 to Ptr)` case
  auto constValue = dyn_cast<llvm::Constant>(arg);
  if (auto constExpr = dyn_cast<llvm::ConstantExpr>(constValue))
    if (constExpr->getOpcode() == llvm::Instruction::IntToPtr)
      if (auto constInt = dyn_cast<llvm::ConstantInt>(constExpr->getOperand(0)))
        ret = constInt->getZExtValue();
  return ret;
}

#define CHECK_RANGE(_check_var, _limit_var)                                    \
  do {                                                                         \
    if (_check_var >= _limit_var) {                                            \
      llvm::errs() << #_check_var << " [" << _check_var                        \
                   << "] is >= " << #_limit_var << " [" << _limit_var          \
                   << "]\n";                                                   \
      return mlir::failure();                                                  \
    }                                                                          \
  } while (0)

// Perform range checking on qubit and result values. This currently only checks
// QIRMeasureBody and QIRRecordOutput. Checking more than that would
// require comprehending the full list of possible QIS instructions, which is
// not currently feasible.
mlir::LogicalResult verifyQubitAndResultRanges(llvm::Module *llvmModule) {
  std::size_t required_num_qubits = 0;
  std::size_t required_num_results = 0;
  for (llvm::Function &func : *llvmModule) {
    if (func.hasFnAttribute(cudaq::opt::QIREntryPointAttrName)) {
      required_num_qubits = func.getFnAttributeAsParsedInteger(
          cudaq::opt::QIRRequiredQubitsAttrName, required_num_qubits);
      required_num_results = func.getFnAttributeAsParsedInteger(
          cudaq::opt::QIRRequiredResultsAttrName, required_num_results);
      break; // no need to keep looking
    }
  }
  for (llvm::Function &func : *llvmModule) {
    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : block) {
        if (auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst)) {
          if (auto func = callInst->getCalledFunction()) {
            // All results must be in range for output recording functions
            if (func->getName() == cudaq::opt::QIRRecordOutput) {
              auto result = getArgAsInteger(callInst->getArgOperand(0));
              CHECK_RANGE(result, required_num_results);
            }
            // All qubits and results must be in range for measurements
            else if (func->getName() == cudaq::opt::QIRMeasureBody) {
              auto qubit = getArgAsInteger(callInst->getArgOperand(0));
              auto result = getArgAsInteger(callInst->getArgOperand(1));
              CHECK_RANGE(qubit, required_num_qubits);
              CHECK_RANGE(result, required_num_results);
            }
          }
        }
      }
    }
  }
  return mlir::success();
}

// Verify that only the allowed LLVM instructions are present
mlir::LogicalResult verifyLLVMInstructions(llvm::Module *llvmModule,
                                           bool isBaseProfile,
                                           bool integerComputations,
                                           bool floatComputations) {
  bool isAdaptiveProfile = !isBaseProfile;
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        // Only specific instructions are allowed at the top level, depending on
        // the specific profile
        bool isValidBaseProfileInstruction =
            llvm::isa<llvm::CallBase>(inst) ||
            llvm::isa<llvm::BranchInst>(inst) ||
            llvm::isa<llvm::ReturnInst>(inst);
        // By default, the adaptive profile supports the same set of
        // instructions as the base profile. Extra/optional
        // instructions/capabilities can be enabled in the target config. For
        // example, `qir-adaptive[int_computations]` to allow integer
        // computation instructions.
        bool isValidAdaptiveProfileInstruction = isValidBaseProfileInstruction;
        if (isBaseProfile && !isValidBaseProfileInstruction) {
          llvm::errs()
              << "error - invalid instruction found in base QIR profile: "
              << inst << '\n';
          return mlir::failure();
        } else if (isAdaptiveProfile && !isValidAdaptiveProfileInstruction) {
          // Not a valid adaptive profile instruction
          // Check if it's in the extended instruction set

          const bool isValidIntExtension =
              integerComputations && isValidIntegerArithmeticInstruction(inst);

          const bool isValidFloatExtension =
              floatComputations && isValidFloatingArithmeticInstruction(inst);

          const bool isValidOutputCall = isValidOutputCallInstruction(inst);
          if (!isValidIntExtension && !isValidFloatExtension &&
              !isValidOutputCall) {
            llvm::errs()
                << "error - invalid instruction found in adaptive QIR profile: "
                << inst << '\n';
            return mlir::failure();
          }
        }

        // Only inttoptr and getelementptr instructions are present as inlined
        // call argument operations. These instructions may not be present
        // unless they inlined call argument operations.
        if (auto *call = dyn_cast<llvm::CallBase>(&inst)) {
          for (auto &arg : call->args()) {
            auto constExpr = llvm::dyn_cast_or_null<llvm::ConstantExpr>(arg);
            if (constExpr &&
                constExpr->getOpcode() != llvm::Instruction::GetElementPtr &&
                constExpr->getOpcode() != llvm::Instruction::IntToPtr) {
              llvm::errs()
                  << "error - invalid instruction found in QIR profile: "
                  << *constExpr << '\n';
              return mlir::failure();
            }
          }
        }
      }
  return mlir::success();
}

/// @brief Function to lower MLIR to a specific QIR profile
/// @param op MLIR operation
/// @param output Output stream
/// @param additionalPasses Additional passes to run at the end
/// @param printIR Print IR to `stderr`
/// @param printIntermediateMLIR Print IR in between each pass
mlir::LogicalResult
qirProfileTranslationFunction(const char *qirProfile, mlir::Operation *op,
                              llvm::raw_string_ostream &output,
                              const std::string &additionalPasses, bool printIR,
                              bool printIntermediateMLIR, bool printStats) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "qirProfileTranslationFunction");

  const std::uint32_t qir_major_version = 1;
  const std::uint32_t qir_minor_version = 0;

  const bool isAdaptiveProfile =
      std::string{qirProfile}.starts_with("qir-adaptive");
  const bool supportIntegerComputations =
      (std::string{qirProfile} == "qir-adaptive-i" ||
       std::string{qirProfile} == "qir-adaptive-if");
  const bool supportFloatComputations =
      (std::string{qirProfile} == "qir-adaptive-f" ||
       std::string{qirProfile} == "qir-adaptive-if");
  const bool isBaseProfile = !isAdaptiveProfile;

  auto context = op->getContext();
  mlir::PassManager pm(context);
  if (printIntermediateMLIR)
    pm.enableIRPrinting();
  if (printStats)
    pm.enableStatistics();
  std::string errMsg;
  llvm::raw_string_ostream errOs(errMsg);
  bool containsWireSet =
      op->walk<mlir::WalkOrder::PreOrder>([](quake::WireSetOp wireSetOp) {
          return mlir::WalkResult::interrupt();
        }).wasInterrupted();

  const std::string rootQirProfileName =
      isAdaptiveProfile ? "qir-adaptive" : qirProfile;
  if (containsWireSet)
    cudaq::opt::addWiresetToProfileQIRPipeline(pm, rootQirProfileName);
  else
    cudaq::opt::addPipelineConvertToQIR(pm, rootQirProfileName);

  // Add additional passes if necessary
  if (!additionalPasses.empty() &&
      failed(parsePassPipeline(additionalPasses, pm, errOs)))
    return mlir::failure();
  mlir::DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run
  if (failed(pm.run(op)))
    return mlir::failure();
  timingScope.stop();

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  llvmContext->setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(op, *llvmContext);

  // Apply required attributes for the Base Profile
  applyWriteOnlyAttributes(llvmModule.get());

  // Add required module flags for the Base Profile
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRMajorVersionFlagName,
                            qir_major_version);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Max,
                            cudaq::opt::QIRMinorVersionFlagName,
                            qir_minor_version);
  auto falseValue =
      llvm::ConstantInt::getFalse(llvm::Type::getInt1Ty(*llvmContext));
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicQubitsManagementFlagName,
                            falseValue);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicResultManagementFlagName,
                            falseValue);

  // Note: optimizeLLVM is the one that is setting nonnull attributes on
  // the @__quantum__rt__result_record_output calls.
  cudaq::optimizeLLVM(llvmModule.get());
  if (!cudaq::setupTargetTriple(llvmModule.get()))
    throw std::runtime_error("Failed to setup the llvm module target triple.");

  if (isAdaptiveProfile)
    applyQIRAdaptiveCapabilitiesAttributes(llvmModule.get());

  // PyQIR currently requires named blocks. It's not clear if blocks can share
  // names across functions, so we are being conservative by giving every block
  // in the module a unique name for now.
  int blockCounter = 0;
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      if (!block.hasName())
        block.setName(std::to_string(blockCounter++));

  if (printIR)
    llvm::errs() << *llvmModule;

  if (failed(verifyOutputRecordingFunctions(llvmModule.get(), isBaseProfile)))
    return mlir::failure();

  if (isBaseProfile &&
      failed(verifyBaseProfileMeasurementOrdering(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyQubitAndResultRanges(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyLLVMInstructions(llvmModule.get(), isBaseProfile,
                                    supportIntegerComputations,
                                    supportFloatComputations)))
    return mlir::failure();

  // Map the LLVM Module to Bitcode that can be submitted
  llvm::SmallString<1024> bitCodeMem;
  llvm::raw_svector_ostream os(bitCodeMem);
  llvm::WriteBitcodeToFile(*llvmModule, os);
  output << llvm::encodeBase64(bitCodeMem.str());
  return mlir::success();
}

void registerToQIRTranslation() {
#define CREATE_QIR_REGISTRATION(_regName, _profile)                            \
  cudaq::TranslateFromMLIRRegistration _regName(                               \
      _profile, "translate from quake to " _profile,                           \
      [](mlir::Operation *op, llvm::raw_string_ostream &output,                \
         const std::string &additionalPasses, bool printIR,                    \
         bool printIntermediateMLIR, bool printStats) {                        \
        return qirProfileTranslationFunction(                                  \
            _profile, op, output, additionalPasses, printIR,                   \
            printIntermediateMLIR, printStats);                                \
      })

  // Base Profile and Adaptive Profile are very similar, so they use the same
  // overall function. We just pass a string to it to tell the function which
  // one is being done.
  // The adaptive profile can support optional integer and/or floating point
  // computations capabilities. These additional capabilities will determine how
  // we validate the output QIR.
  CREATE_QIR_REGISTRATION(regBase, "qir-base");
  // Base adaptive profile
  CREATE_QIR_REGISTRATION(regAdaptive, "qir-adaptive");
  // Adaptive with integer computations
  CREATE_QIR_REGISTRATION(regAdaptiveI, "qir-adaptive-i");
  // Adaptive with floating point computations
  // FIXME: not sure if there is a platform with floating point support but not
  // integer. We just have it here for completeness.
  CREATE_QIR_REGISTRATION(regAdaptiveF, "qir-adaptive-f");
  // Adaptive with integer and floating point computations
  CREATE_QIR_REGISTRATION(regAdaptiveIF, "qir-adaptive-if");
}

void registerToOpenQASMTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "qasm2", "translate from quake to openQASM 2.0",
      [](mlir::Operation *op, llvm::raw_string_ostream &output,
         const std::string &additionalPasses, bool printIR,
         bool printIntermediateMLIR, bool printStats) {
        ScopedTraceWithContext(cudaq::TIMING_JIT, "qasm2 translation");
        mlir::PassManager pm(op->getContext());
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
        if (printStats)
          pm.enableStatistics();
        cudaq::opt::addPipelineTranslateToOpenQASM(pm);
        mlir::DefaultTimingManager tm;
        tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
        auto timingScope = tm.getRootScope(); // starts the timer
        pm.enableTiming(timingScope);         // do this right before pm.run
        if (failed(pm.run(op)))
          throw std::runtime_error("code generation failed.");
        timingScope.stop();
        auto passed = cudaq::translateToOpenQASM(op, output);
        if (printIR) {
          if (succeeded(passed))
            llvm::errs() << output.str();
          else
            llvm::errs() << "failed to create OpenQASM file.";
        }
        return passed;
      });
}

void registerToIQMJsonTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "iqm", "translate from quake to IQM's json format",
      [](mlir::Operation *op, llvm::raw_string_ostream &output,
         const std::string &additionalPasses, bool printIR,
         bool printIntermediateMLIR, bool printStats) {
        ScopedTraceWithContext(cudaq::TIMING_JIT, "iqm translation");
        mlir::PassManager pm(op->getContext());
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
        if (printStats)
          pm.enableStatistics();
        cudaq::opt::addPipelineTranslateToIQMJson(pm);
        mlir::DefaultTimingManager tm;
        tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
        auto timingScope = tm.getRootScope(); // starts the timer
        pm.enableTiming(timingScope);         // do this right before pm.run
        if (failed(pm.run(op)))
          throw std::runtime_error("code generation failed.");
        timingScope.stop();
        auto passed = cudaq::translateToIQMJson(op, output);
        if (printIR) {
          if (succeeded(passed))
            llvm::errs() << output.str();
          else
            llvm::errs() << "failed to create IQM json file.";
        }
        return passed;
      });
}

void insertSetupAndCleanupOperations(mlir::Operation *module) {
  mlir::OpBuilder modBuilder(module);
  auto *context = module->getContext();
  auto arrayQubitTy = cudaq::opt::getArrayType(context);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(context);
  auto boolTy = modBuilder.getI1Type();
  mlir::FlatSymbolRefAttr allocateSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRArrayQubitAllocateArray, arrayQubitTy,
          {modBuilder.getI64Type()}, dyn_cast<mlir::ModuleOp>(module));
  mlir::FlatSymbolRefAttr releaseSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRArrayQubitReleaseArray, {voidTy}, {arrayQubitTy},
          dyn_cast<mlir::ModuleOp>(module));
  mlir::FlatSymbolRefAttr isDynamicSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRisDynamicQubitManagement, {boolTy}, {},
          dyn_cast<mlir::ModuleOp>(module));
  mlir::FlatSymbolRefAttr setDynamicSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRsetDynamicQubitManagement, {voidTy}, {boolTy},
          dyn_cast<mlir::ModuleOp>(module));
  mlir::FlatSymbolRefAttr clearResultMapsSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRClearResultMaps, {voidTy}, {},
          dyn_cast<mlir::ModuleOp>(module));

  // Iterate through all operations in the ModuleOp
  mlir::SmallVector<mlir::LLVM::LLVMFuncOp> funcs;
  module->walk([&](mlir::LLVM::LLVMFuncOp func) { funcs.push_back(func); });
  for (auto &func : funcs) {
    if (!func->hasAttr(cudaq::entryPointAttrName))
      continue;
    std::int64_t num_qubits = -1;
    if (auto requiredQubits = func->getAttrOfType<mlir::StringAttr>(
            cudaq::opt::QIRRequiredQubitsAttrName))
      requiredQubits.strref().getAsInteger(10, num_qubits);

    // Further processing on funcOp if needed
    auto &blocks = func.getBlocks();
    if (blocks.size() < 1 || num_qubits < 0)
      continue;

    mlir::Block &block = *blocks.begin();
    mlir::OpBuilder builder(&block, block.begin());
    auto loc = builder.getUnknownLoc();

    auto origMode = builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{boolTy}, isDynamicSymbol, mlir::ValueRange{});

    // Create constant op
    auto numQubitsVal =
        cudaq::opt::factory::genLlvmI64Constant(loc, builder, num_qubits);
    auto falseVal = builder.create<mlir::LLVM::ConstantOp>(
        loc, boolTy, builder.getI16IntegerAttr(false));

    // Invoke allocate function with constant op
    auto qubitAlloc = builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{arrayQubitTy}, allocateSymbol,
        mlir::ValueRange{numQubitsVal.getResult()});
    builder.create<mlir::LLVM::CallOp>(loc, mlir::TypeRange{voidTy},
                                       setDynamicSymbol,
                                       mlir::ValueRange{falseVal.getResult()});

    // At the end of the function, deallocate the qubits and restore the
    // simulator state.
    builder.setInsertionPoint(std::prev(blocks.end())->getTerminator());
    builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{voidTy}, releaseSymbol,
        mlir::ValueRange{qubitAlloc.getResult()});
    builder.create<mlir::LLVM::CallOp>(loc, mlir::TypeRange{voidTy},
                                       setDynamicSymbol,
                                       mlir::ValueRange{origMode.getResult()});
    builder.create<mlir::LLVM::CallOp>(loc, mlir::TypeRange{voidTy},
                                       clearResultMapsSymbol,
                                       mlir::ValueRange{});
  }
}

mlir::ExecutionEngine *createQIRJITEngine(mlir::ModuleOp &moduleOp,
                                          llvm::StringRef convertTo) {
  // The "fast" instruction selection compilation algorithm is actually very
  // slow for large quantum circuits. Disable that here. Revisit this
  // decision by testing large UCCSD circuits if jitCodeGenOptLevel is changed
  // in the future. Also note that llvm::TargetMachine::setFastIsel() and
  // setO0WantsFastISel() do not retain their values in our current version of
  // LLVM. This use of LLVM command line parameters could be changed if the LLVM
  // JIT ever supports the TargetMachine options in the future.
  ScopedTraceWithContext(cudaq::TIMING_JIT, "createQIRJITEngine");
  const char *argv[] = {"", "-fast-isel=0", nullptr};
  llvm::cl::ParseCommandLineOptions(2, argv);

  mlir::ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  opts.llvmModuleBuilder =
      [convertTo = convertTo.str()](
          mlir::Operation *module,
          llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    ScopedTraceWithContext(cudaq::TIMING_JIT,
                           "createQIRJITEngine::llvmModuleBuilder");
    llvmContext.setOpaquePointers(false);

    auto *context = module->getContext();
    mlir::PassManager pm(context);
    std::string errMsg;
    llvm::raw_string_ostream errOs(errMsg);

    bool containsWireSet =
        module
            ->walk<mlir::WalkOrder::PreOrder>([](quake::WireSetOp wireSetOp) {
              return mlir::WalkResult::interrupt();
            })
            .wasInterrupted();

    // Even though we're not lowering all the way to a real QIR profile for this
    // emulated path, we need to pass in the `convertTo` in order to mimic what
    // the non-emulated path would do.
    if (containsWireSet)
      cudaq::opt::addWiresetToProfileQIRPipeline(pm, convertTo);
    else
      cudaq::opt::commonPipelineConvertToQIR(pm, "qir", convertTo);

    auto enablePrintMLIREachPass =
        getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
    if (enablePrintMLIREachPass) {
      module->getContext()->disableMultithreading();
      pm.enableIRPrinting();
    }

    mlir::DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run
    if (failed(pm.run(module)))
      throw std::runtime_error(
          "[createQIRJITEngine] Lowering to QIR for remote emulation failed.");
    timingScope.stop();

    // Insert necessary calls to qubit allocations and qubit releases if the
    // original module contained WireSetOp's. This is required because the
    // output of the above pipeline will produce IR that uses statically
    // allocated qubit IDs in that case, and the simulator needs these
    // additional calls in order to operate properly.
    if (containsWireSet)
      insertSetupAndCleanupOperations(module);

    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
      throw std::runtime_error(
          "[createQIRJITEngine] Lowering to LLVM IR failed.");

    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
    return llvmModule;
  };

  auto jitOrError = mlir::ExecutionEngine::create(moduleOp, opts);
  assert(!!jitOrError && "ExecutionEngine creation failed.");
  return jitOrError.get().release();
}

} // namespace cudaq
