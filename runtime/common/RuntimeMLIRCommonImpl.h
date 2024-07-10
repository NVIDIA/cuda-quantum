/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Logger.h"
#include "Timing.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/IQMJsonEmitter.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
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
  auto optPipeline = mlir::makeOptimizingTransformer(
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
          bool isIrreversible = calledFunc->hasFnAttribute("irreversible");
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
mlir::LogicalResult verifyOutputRecordingFunctions(llvm::Module *llvmModule) {
  for (llvm::Function &func : *llvmModule) {
    std::set<std::string> outputList;
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);
        auto func = callInst ? callInst->getCalledFunction() : nullptr;
        // All call arguments must be constants
        if (func && failed(verifyConstArguments(callInst)))
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
    if (func.hasFnAttribute("entry_point")) {
      required_num_qubits = func.getFnAttributeAsParsedInteger(
          "requiredQubits", required_num_qubits);
      required_num_results = func.getFnAttributeAsParsedInteger(
          "requiredResults", required_num_results);
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
                                           bool isBaseProfile) {
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
        // Note: there is an outstanding question about the adaptive profile
        // with respect to `switch` and `select` instructions. They are
        // currently described as "optional" in the spec, but there is no way to
        // specify their presence via module flags. So to be cautious, for now
        // we will assume they are not allowed in cuda-quantum programs.
        bool isValidAdaptiveProfileInstruction = isValidBaseProfileInstruction;
        // bool isValidAdaptiveProfileInstruction =
        //     isValidBaseProfileInstruction ||
        //     llvm::isa<llvm::SwitchInst>(inst) ||
        //     llvm::isa<llvm::SelectInst>(inst);
        if (isBaseProfile && !isValidBaseProfileInstruction) {
          llvm::errs() << "error - invalid instruction found: " << inst << '\n';
          return mlir::failure();
        } else if (isAdaptiveProfile && !isValidAdaptiveProfileInstruction) {
          llvm::errs() << "error - invalid instruction found: " << inst << '\n';
          return mlir::failure();
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
                constExpr->getOpcode() != llvm::Instruction::IntToPtr) {
              llvm::errs() << "error - invalid instruction found: "
                           << *constExpr << '\n';
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
mlir::LogicalResult
qirProfileTranslationFunction(const char *qirProfile, mlir::Operation *op,
                              llvm::raw_string_ostream &output,
                              const std::string &additionalPasses, bool printIR,
                              bool printIntermediateMLIR) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "qirProfileTranslationFunction");

  const std::uint32_t qir_major_version = 1;
  const std::uint32_t qir_minor_version = 0;

  const bool isAdaptiveProfile = std::string{qirProfile} == "qir-adaptive";
  const bool isBaseProfile = !isAdaptiveProfile;

  auto context = op->getContext();
  mlir::PassManager pm(context);
  if (printIntermediateMLIR)
    pm.enableIRPrinting();
  std::string errMsg;
  llvm::raw_string_ostream errOs(errMsg);
  cudaq::opt::addPipelineConvertToQIR(pm, qirProfile);
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
                            "qir_major_version", qir_major_version);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Max,
                            "qir_minor_version", qir_minor_version);
  auto falseValue =
      llvm::ConstantInt::getFalse(llvm::Type::getInt1Ty(*llvmContext));
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            "dynamic_qubit_management", falseValue);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            "dynamic_result_management", falseValue);
  if (isAdaptiveProfile) {
    auto trueValue =
        llvm::ConstantInt::getTrue(llvm::Type::getInt1Ty(*llvmContext));
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "qubit_resetting", trueValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "classical_ints", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "classical_floats", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "classical_fixed_points", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "user_functions", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "dynamic_float_args", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "extern_functions", falseValue);
    llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                              "backwards_branching", falseValue);
  }

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

  if (failed(verifyOutputRecordingFunctions(llvmModule.get())))
    return mlir::failure();

  if (isBaseProfile &&
      failed(verifyBaseProfileMeasurementOrdering(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyQubitAndResultRanges(llvmModule.get())))
    return mlir::failure();

  if (failed(verifyLLVMInstructions(llvmModule.get(), isBaseProfile)))
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
         bool printIntermediateMLIR) {                                         \
        return qirProfileTranslationFunction(_profile, op, output,             \
                                             additionalPasses, printIR,        \
                                             printIntermediateMLIR);           \
      })

  // Base Profile and Adaptive Profile are very similar, so they use the same
  // overall function. We just pass a string to it to tell the function which
  // one is being done.
  CREATE_QIR_REGISTRATION(regBase, "qir-base");
  CREATE_QIR_REGISTRATION(regAdaptive, "qir-adaptive");
}

void registerToOpenQASMTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "qasm2", "translate from quake to openQASM 2.0",
      [](mlir::Operation *op, llvm::raw_string_ostream &output,
         const std::string &additionalPasses, bool printIR,
         bool printIntermediateMLIR) {
        ScopedTraceWithContext(cudaq::TIMING_JIT, "qasm2 translation");
        mlir::PassManager pm(op->getContext());
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
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
         bool printIntermediateMLIR) {
        ScopedTraceWithContext(cudaq::TIMING_JIT, "iqm translation");
        mlir::PassManager pm(op->getContext());
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
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
    // Even though we're not lowering all the way to a real QIR profile for this
    // emulated path, we need to pass in the `convertTo` in order to mimic what
    // the non-emulated path would do.
    cudaq::opt::commonPipelineConvertToQIR(pm, convertTo);
    mlir::DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run
    if (failed(pm.run(module)))
      throw std::runtime_error(
          "[createQIRJITEngine] Lowering to QIR for remote emulation failed.");
    timingScope.stop();
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
