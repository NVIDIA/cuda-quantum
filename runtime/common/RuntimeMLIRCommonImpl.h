/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CodeGenConfig.h"
#include "Environment.h"
#include "Logger.h"
#include "Timing.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/OptUtils.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
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

void initializeLangMLIR();

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
    if (func.hasFnAttribute("entry_point"))
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

static bool isValidIntegerArithmeticInstruction(llvm::Instruction &inst) {
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

// Once a call to a function with irreversible attribute is seen, no more calls
// to reversible functions are allowed. This is somewhat of an implied
// specification because the specification describes the program in terms of 4
// sequential blocks. The 2nd block contains reversible operations, and the 3rd
// block contains irreversible operations (measurements), and the blocks may not
// overlap.
// Reference:
// https://github.com/qir-alliance/qir-spec/blob/684b17b/specification/profiles/Base_Profile.md#L196
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
          bool isIrreversible = calledFunc->hasFnAttribute("irreversible");
          bool isReversible = !isIrreversible;
          bool isOutputFunction =
              (funcName == cudaq::opt::QIRRecordOutput ||
               funcName == cudaq::opt::QIRArrayRecordOutput);
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
  if (auto constExpr = dyn_cast_if_present<llvm::ConstantExpr>(constValue))
    if (constExpr->getOpcode() == llvm::Instruction::IntToPtr)
      if (auto constInt =
              dyn_cast_if_present<llvm::ConstantInt>(constExpr->getOperand(0)))
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
    if (func.hasFnAttribute("entry_point")) {
      constexpr auto NotFound = std::numeric_limits<std::uint64_t>::max();
      required_num_qubits = func.getFnAttributeAsParsedInteger(
          cudaq::opt::qir0_1::RequiredQubitsAttrName, NotFound);
      if (required_num_qubits == NotFound)
        required_num_qubits = func.getFnAttributeAsParsedInteger(
            cudaq::opt::qir1_0::RequiredQubitsAttrName, 0);
      required_num_results = func.getFnAttributeAsParsedInteger(
          cudaq::opt::qir0_1::RequiredResultsAttrName, NotFound);
      if (required_num_results == NotFound)
        required_num_results = func.getFnAttributeAsParsedInteger(
            cudaq::opt::qir1_0::RequiredResultsAttrName, 0);
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

/// Filter out code patterns that do not meet the accepted QIR specification for
/// a particular target. The patterns are selectable via environment variables.
/// Note that no analysis is used and this simply drops code on the floor. As
/// such, the code may not function correctly nor as expected.
static mlir::LogicalResult filterSpecificCodePatterns(llvm::Module *llvmModule,
                                                      CodeGenConfig &config) {
  bool erasePatterns = config.outputLog;
  bool eraseStackBounding = config.eraseStackBounding;
  bool eraseResultRecordCalls = config.eraseRecordCalls;

  if (erasePatterns || eraseStackBounding || eraseResultRecordCalls) {
    llvm::SmallVector<llvm::Instruction *> eraseInst;
    for (llvm::Function &func : *llvmModule)
      for (llvm::BasicBlock &block : func)
        for (llvm::Instruction &inst : block)
          if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
            auto *calledFunc = call->getCalledFunction();
            auto name = calledFunc->getGlobalIdentifier();
            if (eraseStackBounding && calledFunc->isIntrinsic() &&
                (name == cudaq::llvmStackSave ||
                 name == cudaq::llvmStackRestore))
              eraseInst.push_back(&inst);
            if (eraseResultRecordCalls && name == cudaq::opt::QIRRecordOutput)
              eraseInst.push_back(&inst);
          }
    for (auto *insn : eraseInst) {
      if (insn->hasNUsesOrMore(1))
        insn->replaceAllUsesWith(llvm::UndefValue::get(insn->getType()));
      insn->eraseFromParent();
    }
  }
  return mlir::success();
}

/// Verify that only LLVM instructions allowed by the QIR specification per the
/// selected profile, version, and extensions are present.
mlir::LogicalResult verifyLLVMInstructions(llvm::Module *llvmModule,
                                           CodeGenConfig &config) {

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
        // example, `qir-adaptive:0.1:int_computations` to allow integer
        // computation instructions.
        bool isValidAdaptiveProfileInstruction = isValidBaseProfileInstruction;
        if (config.isBaseProfile && !isValidBaseProfileInstruction) {
          llvm::errs() << "QIR verification error - invalid instruction found: "
                       << inst << " (base profile)\n";
          if (!config.allowAllInstructions)
            return mlir::failure();
        } else if (config.isAdaptiveProfile &&
                   !isValidAdaptiveProfileInstruction) {
          // Not a valid adaptive profile instruction
          // Check if it's in the extended instruction set
          const bool isValidIntExtension =
              config.integerComputations &&
              isValidIntegerArithmeticInstruction(inst);

          const bool isValidFloatExtension =
              config.floatComputations &&
              isValidFloatingArithmeticInstruction(inst);

          const bool isValidOutputCall = isValidOutputCallInstruction(inst);
          if (!isValidIntExtension && !isValidFloatExtension &&
              !isValidOutputCall) {
            llvm::errs()
                << "QIR verification error - invalid instruction found: "
                << inst << " (adaptive profile)\n";
            if (!config.allowAllInstructions)
              return mlir::failure();
          }
        }
        // Only inttoptr and getelementptr instructions are present as inlined
        // call argument operations. These instructions may not be present
        // unless they inlined call argument operations.
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
              if (!config.allowAllInstructions)
                return mlir::failure();
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
mlir::LogicalResult qirProfileTranslationFunction(
    const std::string &qirProfile, mlir::Operation *op,
    llvm::raw_string_ostream &output, const std::string &additionalPasses,
    bool printIR, bool printIntermediateMLIR, bool printStats) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "qirProfileTranslationFunction");

  auto config = parseCodeGenTranslation(qirProfile);
  if (!config.isQIRProfile)
    throw std::runtime_error(cudaq_fmt::format(
        "Unexpected codegen profile while translating to QIR: {}",
        config.profile));

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

  if (containsWireSet)
    cudaq::opt::addWiresetToProfileQIRPipeline(pm, config.profile);
  else
    cudaq::opt::addAOTPipelineConvertToQIR(pm, qirProfile);

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
                            config.qir_major_version);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Max,
                            cudaq::opt::QIRMinorVersionFlagName,
                            config.qir_minor_version);
  auto falseValue =
      llvm::ConstantInt::getFalse(llvm::Type::getInt1Ty(*llvmContext));
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicQubitsManagementFlagName,
                            falseValue);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicResultManagementFlagName,
                            falseValue);
  if (config.isAdaptiveProfile) {
    auto trueValue =
        llvm::ConstantInt::getTrue(llvm::Type::getInt1Ty(*llvmContext));
    if (config.version == QirVersion::version_0_1) {
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::QubitResettingFlagName,
                                trueValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ClassicalIntsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ClassicalFloatsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(
          llvm::Module::ModFlagBehavior::Error,
          cudaq::opt::qir0_1::ClassicalFixedPointsFlagName, falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::UserFunctionsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::DynamicFloatArgsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ExternFunctionsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::BackwardsBranchingFlagName,
                                falseValue);
    } else {
      // Note: hopefully all QIR versions after 0.1 will start to converge on
      // using the same sets of flags and flag names.
      if (config.integerComputations) {
        llvm::Constant *intPrecisionValue =
            llvm::ConstantDataArray::getString(*llvmContext, "i64", false);
        llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                  cudaq::opt::qir1_0::IntComputationsFlagName,
                                  intPrecisionValue);
      }
      if (config.floatComputations) {
        llvm::Constant *floatPrecisionValue =
            llvm::ConstantDataArray::getString(*llvmContext, "f64", false);
        llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                  cudaq::opt::qir1_0::FloatComputationsFlagName,
                                  floatPrecisionValue);
      }
      auto backwardsBranchingValue = llvm::ConstantInt::getIntegerValue(
          llvm::Type::getIntNTy(*llvmContext, 2), llvm::APInt(2, 0, false));
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir1_0::BackwardsBranchingFlagName,
                                backwardsBranchingValue);
    }
  }

  // There are certain function calls that may be produced that we want to drop
  // on the floor instead of passing to the QIR consumer.
  if (failed(filterSpecificCodePatterns(llvmModule.get(), config)))
    return mlir::failure();

  // Note: optimizeLLVM is the one that is setting nonnull attributes on
  // the @__quantum__rt__result_record_output calls.
  cudaq::optimizeLLVM(llvmModule.get());
  if (!cudaq::setupTargetTriple(llvmModule.get()))
    throw std::runtime_error("Failed to setup the llvm module target triple.");

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

  if (failed(verifyOutputRecordingFunctions(llvmModule.get(),
                                            config.isBaseProfile)))
    return mlir::failure();

  if (config.isBaseProfile &&
      failed(verifyBaseProfileMeasurementOrdering(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyQubitAndResultRanges(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyLLVMInstructions(llvmModule.get(), config)))
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
      [](mlir::Operation *op, const std::string &transportTriple,              \
         llvm::raw_string_ostream &output,                                     \
         const std::string &additionalPasses, bool printIR,                    \
         bool printIntermediateMLIR, bool printStats) {                        \
        return qirProfileTranslationFunction(                                  \
            transportTriple, op, output, additionalPasses, printIR,            \
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
            cudaq::opt::qir0_1::RequiredQubitsAttrName))
      requiredQubits.strref().getAsInteger(10, num_qubits);
    else if (auto requiredQubits = func->getAttrOfType<mlir::StringAttr>(
                 cudaq::opt::qir1_0::RequiredQubitsAttrName))
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
      cudaq::opt::addAOTPipelineConvertToQIR(pm);

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
