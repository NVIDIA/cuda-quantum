/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/IQM/IQMJsonEmitter.h"
#include "cudaq/Target/OpenQASM/OpenQASMEmitter.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace cudaq {
static bool mlirLLVMInitialized = false;

static llvm::StringMap<cudaq::Translation> &getTranslationRegistry() {
  static llvm::StringMap<cudaq::Translation> translationBundle;
  return translationBundle;
}
cudaq::Translation &getTranslation(StringRef name) {
  auto &registry = getTranslationRegistry();
  if (!registry.count(name))
    throw std::runtime_error("Invalid IR Translation (" + name.str() + ").");
  return registry[name];
}

static void registerTranslation(StringRef name, StringRef description,
                                const TranslateFromMLIRFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    return;
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = cudaq::Translation(function, description);
}

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function) {
  registerTranslation(name, description, function);
}

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
  bool enableOpt = true;
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(module))
    throw std::runtime_error("Failed to optimize LLVM IR ");
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
mlir::LogicalResult verifyMeasurementOrdering(llvm::Module *llvmModule) {
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
              funcName == cudaq::opt::QIRBaseProfileRecordOutput;
          if (isReversible && !isOutputFunction && irreversibleSeenYet) {
            llvm::errs() << "error: reversible function " << funcName
                         << " came after irreversible function\n";
            return failure();
          }
          if (isIrreversible)
            irreversibleSeenYet = true;
        }
      }
  return success();
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
                     << cudaq::opt::QIRBaseProfileRecordOutput << " function\n";
        return failure();
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
              return failure();
            }
          }
        }
      }
    }

    iArg++;
  }
  return success();
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
      return failure();
    }
    iArg++;
  }
  return success();
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
          return failure();
        // If it's an output function, do additional verification
        if (func && func->getName() == cudaq::opt::QIRBaseProfileRecordOutput)
          if (failed(verifyOutputCalls(callInst, outputList)))
            return failure();
      }
  }
  return success();
}

// Verify that only the allowed LLVM instructions are present
mlir::LogicalResult verifyLLVMInstructions(llvm::Module *llvmModule) {
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        // Only call, br, and ret instructions are allowed at the top level.
        if (!llvm::isa<llvm::CallBase>(inst) &&
            !llvm::isa<llvm::BranchInst>(inst) &&
            !llvm::isa<llvm::ReturnInst>(inst)) {
          llvm::errs() << "error - invalid instruction found: " << inst << '\n';
          return failure();
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
              return failure();
            }
          }
      }
  return success();
}

void registerToQIRTranslation() {
  const uint32_t qir_major_version = 1;
  const uint32_t qir_minor_version = 0;

  cudaq::TranslateFromMLIRRegistration reg(
      "qir", "translate from quake to qir base profile",
      [](Operation *op, llvm::raw_string_ostream &output, bool printIR,
         bool printIntermediateMLIR) {
        auto context = op->getContext();
        PassManager pm(context);
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
        std::string errMsg;
        llvm::raw_string_ostream errOs(errMsg);
        auto qirBasePipelineConfig =
            "func.func(combine-quantum-alloc),canonicalize,cse,quake-to-qir,"
            "base-profile-pipeline";
        if (failed(parsePassPipeline(qirBasePipelineConfig, pm, errOs)))
          return failure();
        if (failed(pm.run(op)))
          return failure();

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

        // Note: optimizeLLVM is the one that is setting nonnull attributes on
        // the @__quantum__rt__result_record_output calls.
        cudaq::optimizeLLVM(llvmModule.get());
        if (!cudaq::setupTargetTriple(llvmModule.get()))
          throw std::runtime_error(
              "Failed to setup the llvm module target triple.");

        if (printIR)
          llvm::errs() << *llvmModule;

        if (failed(verifyOutputRecordingFunctions(llvmModule.get())))
          return failure();

        if (failed(verifyMeasurementOrdering(llvmModule.get())))
          return failure();

        if (failed(verifyLLVMInstructions(llvmModule.get())))
          return failure();

        // Map the LLVM Module to Bitcode that can be submitted
        llvm::SmallString<1024> bitCodeMem;
        llvm::raw_svector_ostream os(bitCodeMem);
        llvm::WriteBitcodeToFile(*llvmModule, os);
        output << llvm::encodeBase64(bitCodeMem.str());
        return success();
      });
}

void registerToOpenQASMTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "qasm2", "translate from quake to openQASM 2.0",
      [](Operation *op, llvm::raw_string_ostream &output, bool printIR,
         bool printIntermediateMLIR) {
        PassManager pm(op->getContext());
        if (printIntermediateMLIR)
          pm.enableIRPrinting();
        if (failed(pm.run(op)))
          throw std::runtime_error("Lowering failed.");
        auto passed = cudaq::translateToOpenQASM(op, output);
        if (printIR)
          llvm::errs() << output.str();
        return passed;
      });
}

void registerToIQMJsonTranslation() {
  cudaq::TranslateFromMLIRRegistration reg(
      "iqm", "translate from quake to IQM's json format",
      [](Operation *op, llvm::raw_string_ostream &output, bool printIR,
         bool printIntermediateMLIR) {
        auto passed = cudaq::translateToIQMJson(op, output);
        if (printIntermediateMLIR)
          llvm::errs() << "WARNING: printIntermediateMLIR not supported for "
                          "IQM's json format\n";
        if (printIR)
          llvm::errs() << output.str();
        return passed;
      });
}

std::unique_ptr<MLIRContext> initializeMLIR() {
  if (!mlirLLVMInitialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    registerAllPasses();
    cudaq::opt::registerOptCodeGenPasses();
    cudaq::opt::registerOptTransformsPasses();
    registerToQIRTranslation();
    registerToOpenQASMTranslation();
    registerToIQMJsonTranslation();
    cudaq::opt::registerUnrollingPipeline();
    cudaq::opt::registerBaseProfilePipeline();
    cudaq::opt::registerTargetPipelines();
    mlirLLVMInitialized = true;
  }

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, LLVM::LLVMDialect, math::MathDialect,
                  memref::MemRefDialect, quake::QuakeDialect, cc::CCDialect,
                  func::FuncDialect>();
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  registerLLVMDialectTranslation(*context);
  return context;
}

ExecutionEngine *createQIRJITEngine(ModuleOp &moduleOp) {
  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  opts.llvmModuleBuilder =
      [&](Operation *module,
          llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    llvmContext.setOpaquePointers(false);

    auto *context = module->getContext();
    PassManager pm(context);
    std::string errMsg;
    llvm::raw_string_ostream errOs(errMsg);
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createCombineQuantumAllocations());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(cudaq::opt::createConvertToQIRPass());
    if (failed(pm.run(module)))
      throw std::runtime_error(
          "[createQIRJITEngine] Lowering to QIR for remote emulation failed.");
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
      throw std::runtime_error(
          "[createQIRJITEngine] Lowering to LLVM IR failed.");

    ExecutionEngine::setupTargetTriple(llvmModule.get());
    return llvmModule;
  };

  auto jitOrError = ExecutionEngine::create(moduleOp, opts);
  assert(!!jitOrError && "ExecutionEngine creation failed.");
  return jitOrError.get().release();
}

} // namespace cudaq
