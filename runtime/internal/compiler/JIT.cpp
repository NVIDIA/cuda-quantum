/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/JIT.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/Timing.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Verifier/QIRLLVMIRDialect.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <cassert>
#include <cxxabi.h>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>

#define DEBUG_TYPE "cudaq-qpud"

using namespace mlir;

static void insertSetupAndCleanupOperations(Operation *module) {
  OpBuilder modBuilder(module);
  auto *context = module->getContext();
  auto arrayQubitTy = cudaq::cg::getLLVMArrayType(context);
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto boolTy = modBuilder.getI1Type();
  FlatSymbolRefAttr allocateSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRArrayQubitAllocateArray, arrayQubitTy,
          {modBuilder.getI64Type()}, dyn_cast<ModuleOp>(module));
  FlatSymbolRefAttr releaseSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRArrayQubitReleaseArray, {voidTy}, {arrayQubitTy},
          dyn_cast<ModuleOp>(module));
  FlatSymbolRefAttr isDynamicSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRisDynamicQubitManagement, {boolTy}, {},
          dyn_cast<ModuleOp>(module));
  FlatSymbolRefAttr setDynamicSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRsetDynamicQubitManagement, {voidTy}, {boolTy},
          dyn_cast<ModuleOp>(module));
  FlatSymbolRefAttr clearResultMapsSymbol =
      cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::QIRClearResultMaps, {voidTy}, {},
          dyn_cast<ModuleOp>(module));

  // Iterate through all operations in the ModuleOp
  SmallVector<LLVM::LLVMFuncOp> funcs;
  module->walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
  for (auto &func : funcs) {
    if (!func->hasAttr(cudaq::entryPointAttrName))
      continue;
    std::int64_t num_qubits = -1;
    if (auto requiredQubits = func->getAttrOfType<StringAttr>(
            cudaq::opt::qir0_1::RequiredQubitsAttrName))
      requiredQubits.strref().getAsInteger(10, num_qubits);
    else if (auto requiredQubits = func->getAttrOfType<StringAttr>(
                 cudaq::opt::qir1_0::RequiredQubitsAttrName))
      requiredQubits.strref().getAsInteger(10, num_qubits);

    auto &blocks = func.getBlocks();
    if (blocks.size() < 1 || num_qubits < 0)
      continue;

    Block &block = *blocks.begin();
    OpBuilder builder(&block, block.begin());
    auto loc = builder.getUnknownLoc();

    auto origMode =
        mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{boolTy},
                                   isDynamicSymbol, mlir::ValueRange{});

    auto numQubitsVal =
        cudaq::opt::factory::genLlvmI64Constant(loc, builder, num_qubits);
    auto falseVal = mlir::LLVM::ConstantOp::create(
        builder, loc, boolTy, builder.getI16IntegerAttr(false));

    auto qubitAlloc = mlir::LLVM::CallOp::create(
        builder, loc, mlir::TypeRange{arrayQubitTy}, allocateSymbol,
        mlir::ValueRange{numQubitsVal.getResult()});
    mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{voidTy},
                               setDynamicSymbol,
                               mlir::ValueRange{falseVal.getResult()});

    // At the end of the function, deallocate the qubits and restore the
    // simulator state.
    builder.setInsertionPoint(std::prev(blocks.end())->getTerminator());
    mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{voidTy},
                               releaseSymbol,
                               mlir::ValueRange{qubitAlloc.getResult()});
    mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{voidTy},
                               setDynamicSymbol,
                               mlir::ValueRange{origMode.getResult()});
    mlir::LLVM::CallOp::create(builder, loc, mlir::TypeRange{voidTy},
                               clearResultMapsSymbol, mlir::ValueRange{});
  }
}

cudaq::JitEngine
cudaq_internal::compiler::createJITEngine(ModuleOp &moduleOp,
                                          llvm::StringRef convertTo) {
  // The "fast" instruction selection compilation algorithm is actually very
  // slow for large quantum circuits. Disable that here.
  ScopedTraceWithContext(cudaq::TIMING_JIT, "createJITEngine");
  const char *argv[] = {"", "-fast-isel=0", nullptr};
  llvm::cl::ParseCommandLineOptions(2, argv);

  ExecutionEngineOptions opts;
  auto transformerTemp = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.transformer = std::move(transformerTemp);
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
  auto llvmModuleBuilderTemp =
      [convertTo = convertTo.str()](
          Operation *module,
          llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    ScopedTraceWithContext(cudaq::TIMING_JIT,
                           "createJITEngine::llvmModuleBuilder");

    auto *context = module->getContext();
    PassManager pm(context);

    bool containsWireSet =
        module
            ->walk<WalkOrder::PreOrder>([](cudaq::quake::WireSetOp wireSetOp) {
              return WalkResult::interrupt();
            })
            .wasInterrupted();

    // Even though we're not lowering all the way to a real QIR profile for
    // this emulated path, we need to pass in `convertTo` to mimic the
    // non-emulated path.
    std::string profileName;
    if (containsWireSet) {
      profileName = convertTo;
      cudaq::opt::addWiresetToProfileQIRPipeline(pm, profileName);
    } else {
      cudaq::opt::addAOTPipelineConvertToQIR(pm);
    }

    auto enablePrintMLIREachPass =
        cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
    auto disableThreading =
        cudaq::getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", false);
    if (enablePrintMLIREachPass || disableThreading) {
      module->getContext()->disableMultithreading();
      if (enablePrintMLIREachPass)
        pm.enableIRPrinting();
    }

    std::string error_msg;
    DiagnosticEngine &engine = context->getDiagEngine();
    auto handlerId =
        engine.registerHandler([&error_msg](Diagnostic &diag) -> LogicalResult {
          if (diag.getSeverity() == DiagnosticSeverity::Error) {
            error_msg += diag.str();
            return failure(false);
          }
          return failure();
        });

    DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run
    if (failed(cudaq_internal::compiler::runPassManager(pm, module))) {
      engine.eraseHandler(handlerId);
      throw std::runtime_error("[createJITEngine] Lowering to QIR for "
                               "remote emulation failed.\n" +
                               error_msg);
    }
    if (auto mod = dyn_cast<ModuleOp>(module))
      if (failed(cudaq::verifier::checkQIRLLVMIRDialect(mod, profileName)))
        throw std::runtime_error(
            "[createJITEngine] QIR verification failed.\n");

    timingScope.stop();
    engine.eraseHandler(handlerId);

    // Insert necessary calls to qubit allocations and qubit releases if the
    // original module contained WireSetOp's.
    if (containsWireSet)
      insertSetupAndCleanupOperations(module);

    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
      throw std::runtime_error("[createJITEngine] Lowering to LLVM IR failed.");

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (tmBuilderOrError) {
      auto tmOrError = tmBuilderOrError->createTargetMachine();
      if (tmOrError)
        mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
            llvmModule.get(), tmOrError.get().get());
    }
    return llvmModule;
  };
  opts.llvmModuleBuilder = std::move(llvmModuleBuilderTemp);

  auto jitOrError = ExecutionEngine::create(moduleOp, opts);
  assert(!!jitOrError && "ExecutionEngine creation failed.");
  return cudaq::JitEngine(std::move(jitOrError.get()));
}

class cudaq::JitEngine::Impl : public cudaq::JitEngine::Base {
public:
  Impl(std::unique_ptr<ExecutionEngine> jitEngine)
      : jitEngine(std::move(jitEngine)) {
    lookupFn = [this](const std::string &name) -> RawFnPtr {
      auto funcPtr = this->jitEngine->lookup(name);
      if (!funcPtr)
        throw std::runtime_error("Failed looking function up in jitted module");
      return reinterpret_cast<RawFnPtr>(*funcPtr);
    };
    runFn = [this](const std::string &kernelName) {
      auto funcPtr = lookupFn(std::string(cudaq::runtime::cudaqGenPrefixName) +
                              kernelName);
      funcPtr();
    };
  }

  std::size_t getKey() const {
    return reinterpret_cast<std::size_t>(jitEngine.get());
  }

private:
  std::unique_ptr<ExecutionEngine> jitEngine;
};

cudaq::JitEngine::JitEngine(std::unique_ptr<ExecutionEngine> jitEngine)
    : impl(std::make_shared<cudaq::JitEngine::Impl>(std::move(jitEngine))) {}

std::size_t cudaq::JitEngine::getKey() const {
  return static_cast<const Impl *>(impl.get())->getKey();
}
