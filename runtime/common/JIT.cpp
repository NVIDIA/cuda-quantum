/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JIT.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <cxxabi.h>

#define DEBUG_TYPE "cudaq-qpud"

namespace cudaq {
std::unique_ptr<llvm::orc::LLJIT>
invokeWrappedKernel(std::string_view irString, const std::string &entryPointFn,
                    void *args, std::uint64_t argsSize, std::size_t numTimes,
                    std::function<void(std::size_t)> postExecCallback) {

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

  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
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
  for (std::size_t i = 0; i < numTimes; ++i) {
    // Invoke the wrapper with serialized data and the kernel.
    fptrWrapper(args, argsSize, fptr);
    if (postExecCallback) {
      postExecCallback(i);
    }
  }

  return jit;
}
} // namespace cudaq
