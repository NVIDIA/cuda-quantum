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
using namespace cudaq_internal::compiler;
using cudaq::JitEngine;

std::tuple<std::unique_ptr<llvm::orc::LLJIT>, std::function<void()>>
cudaq_internal::compiler::createWrappedKernel(std::string_view irString,
                                              const std::string &entryPointFn,
                                              void *args,
                                              std::uint64_t argsSize) {

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  // Parse bitcode
  llvm::SMDiagnostic Err;
  auto fileBuf = llvm::MemoryBuffer::getMemBufferCopy(irString);
  std::unique_ptr<llvm::Module> llvmModule = llvm::parseIR(*fileBuf, Err, *ctx);
  if (!llvmModule)
    throw "Failed to parse embedded bitcode";

  // Retrieve the symbol names for the kernel and its wrapper.
  const std::pair<std::string, std::string> mangledKernelNames = [&]() {
    const std::string templatedTypeName = [&]() {
      const auto pos = entryPointFn.find_first_of("(");
      return (pos != std::string::npos) ? entryPointFn.substr(pos + 1)
                                        : entryPointFn;
    }();

    const std::string wrappedKernelSymbol =
        "void cudaq::invokeCallableWithSerializedArgs<";

    const std::string funcName = [&]() {
      const auto pos = entryPointFn.find_first_of("(");
      return (pos != std::string::npos) ? entryPointFn.substr(0, pos)
                                        : entryPointFn;
    }();
    std::string mangledKernel, mangledWrapper;

    // Lambda symbols has internal linkage, prevent them from being looked up.
    // Hence, fix the linkage.
    const auto fixUpLinkage = [](auto &func) {
      if (func.hasInternalLinkage()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Change linkage type for symbol " << func.getName()
                   << " internal to external linkage.");
        func.setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      }
    };

    for (auto &func : llvmModule->functions()) {
      auto demangledPtr =
          abi::__cxa_demangle(func.getName().data(), nullptr, nullptr, nullptr);
      if (demangledPtr) {
        std::string demangledName(demangledPtr);
        if (demangledName.rfind(wrappedKernelSymbol, 0) == 0 &&
            demangledName.find(templatedTypeName) != std::string::npos) {
          LLVM_DEBUG(llvm::dbgs() << "Found symbol " << func.getName()
                                  << " for " << wrappedKernelSymbol);
          mangledWrapper = func.getName().str();
          fixUpLinkage(func);
        }
        if (demangledName.rfind(funcName, 0) == 0) {
          LLVM_DEBUG(llvm::dbgs() << "Found symbol " << func.getName()
                                  << " for " << funcName);
          mangledKernel = func.getName().str();
          fixUpLinkage(func);
        }
      }
    }
    return std::make_pair(mangledKernel, mangledWrapper);
  }();

  if (mangledKernelNames.first.empty() || mangledKernelNames.second.empty())
    throw std::runtime_error("Failed to locate symbols from the IR");

  ExecutionEngine::setupTargetTriple(llvmModule.get());
  auto dataLayout = llvmModule->getDataLayout();

  // Create the object layer
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt) {
    auto objectLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
          return std::make_unique<llvm::SectionMemoryManager>();
        });
    llvm::Triple targetTriple(llvm::Twine(llvmModule->getTargetTriple()));
    return objectLayer;
  };

  // Create the LLJIT with the object link layer
  auto jit = llvm::cantFail(
      llvm::orc::LLJITBuilder()
          .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
          .create());

  // Add a ThreadSafemodule to the engine and return.
  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  llvm::cantFail(jit->addIRModule(std::move(tsm)));

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = jit->getMainJITDylib();
  mainJD.addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  // Symbol lookup: kernel and wrapper
  auto kernelSymbolAddr = llvm::cantFail(jit->lookup(mangledKernelNames.first));
  void *fptr = kernelSymbolAddr.toPtr<void *>();
  auto wrapperSymbolAddr =
      llvm::cantFail(jit->lookup(mangledKernelNames.second));
  auto *fptrWrapper =
      wrapperSymbolAddr.toPtr<void (*)(const void *, unsigned long, void *)>();

  auto callable = [args, argsSize, fptr, fptrWrapper]() {
    fptrWrapper(args, argsSize, fptr);
  };
  return std::make_tuple(std::move(jit), callable);
}

namespace {
void insertSetupAndCleanupOperations(Operation *module) {
  OpBuilder modBuilder(module);
  auto *context = module->getContext();
  auto arrayQubitTy = cudaq::opt::getArrayType(context);
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

    auto origMode = builder.create<LLVM::CallOp>(loc, TypeRange{boolTy},
                                                 isDynamicSymbol, ValueRange{});

    auto numQubitsVal =
        cudaq::opt::factory::genLlvmI64Constant(loc, builder, num_qubits);
    auto falseVal = builder.create<LLVM::ConstantOp>(
        loc, boolTy, builder.getI16IntegerAttr(false));

    auto qubitAlloc = builder.create<LLVM::CallOp>(
        loc, TypeRange{arrayQubitTy}, allocateSymbol,
        ValueRange{numQubitsVal.getResult()});
    builder.create<LLVM::CallOp>(loc, TypeRange{voidTy}, setDynamicSymbol,
                                 ValueRange{falseVal.getResult()});

    // At the end of the function, deallocate the qubits and restore the
    // simulator state.
    builder.setInsertionPoint(std::prev(blocks.end())->getTerminator());
    builder.create<LLVM::CallOp>(loc, TypeRange{voidTy}, releaseSymbol,
                                 ValueRange{qubitAlloc.getResult()});
    builder.create<LLVM::CallOp>(loc, TypeRange{voidTy}, setDynamicSymbol,
                                 ValueRange{origMode.getResult()});
    builder.create<LLVM::CallOp>(loc, TypeRange{voidTy}, clearResultMapsSymbol,
                                 ValueRange{});
  }
}
} // namespace

cudaq::JitEngine
cudaq_internal::compiler::createJITEngine(ModuleOp &moduleOp,
                                          llvm::StringRef convertTo) {
  // The "fast" instruction selection compilation algorithm is actually very
  // slow for large quantum circuits. Disable that here.
  ScopedTraceWithContext(cudaq::TIMING_JIT, "createJITEngine");
  const char *argv[] = {"", "-fast-isel=0", nullptr};
  llvm::cl::ParseCommandLineOptions(2, argv);

  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  opts.llvmModuleBuilder =
      [convertTo = convertTo.str()](
          Operation *module,
          llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    ScopedTraceWithContext(cudaq::TIMING_JIT,
                           "createJITEngine::llvmModuleBuilder");
    llvmContext.setOpaquePointers(false);

    auto *context = module->getContext();
    PassManager pm(context);

    bool containsWireSet =
        module
            ->walk<WalkOrder::PreOrder>([](quake::WireSetOp wireSetOp) {
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

    ExecutionEngine::setupTargetTriple(llvmModule.get());
    return llvmModule;
  };

  auto jitOrError = ExecutionEngine::create(moduleOp, opts);
  assert(!!jitOrError && "ExecutionEngine creation failed.");
  return JitEngine(std::move(jitOrError.get()));
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
    : impl(std::make_shared<JitEngine::Impl>(std::move(jitEngine))) {}

std::size_t cudaq::JitEngine::getKey() const {
  return static_cast<const Impl *>(impl.get())->getKey();
}
