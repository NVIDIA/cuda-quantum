/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/InitAllDialects.h"
#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteRESTQPU.h"
#include "common/RuntimeMLIRCommonImpl.h"

// [RFC]:
// The RemoteRESTQPU implementation that is now split across several files needs
// to be examined carefully. What used to be in
// /runtime/cudaq/platform/default/rest/RemoteRESTQPU.cpp is now largely in the
// header /runtime/common/BaseRemoteRESTQPU.h [status 11/18/23], but some
// updates were missed; The updatePassPipeline interface method in the
// ServerHelper, for example, was not invoked at all.
using namespace mlir;

extern "C" void __cudaq_deviceCodeHolderAdd(const char *, const char *);

namespace cudaq {

// We have to reproduce the TranslationRegistry here in this Translation Unit

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

// We cannot use the RemoteRESTQPU since we'll get LLVM / MLIR statically loaded
// twice. We've extracted most of RemoteRESTQPU into BaseRemoteRESTQPU and will
// implement some core functionality here in PyRemoteRESTQPU so we don't load
// twice
class PyRemoteRESTQPU : public cudaq::BaseRemoteRESTQPU {
private:
  /// Creates new context without mlir initialization.
  std::unique_ptr<MLIRContext> createContext() {
    DialectRegistry registry;
    cudaq::opt::registerCodeGenDialect(registry);
    cudaq::registerAllDialects(registry);
    auto context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();

    static bool initOnce = [&] {
      registerToQIRTranslation();
      registerToOpenQASMTranslation();
      registerToIQMJsonTranslation();
      registerLLVMDialectTranslation(*context);
      return true;
    }();
    (void)initOnce;

    return context;
  }

protected:
 std::tuple<mlir::ModuleOp, mlir::MLIRContext *>
extractQuakeCodeAndContext(const std::string &kernelName) override {

    auto contextPtr = createContext();
    MLIRContext* context = contextPtr.get();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, context);
    std::cout << "parsed module" << std::endl;
    m_module->dump();

    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    // Here we have an opportunity to run any passes that are
    // specific to python before the rest of the RemoteRESTQPU workflow
    auto cloned = m_module->clone();
    PassManager pm(cloned.getContext());

    cudaq::opt::addAggressiveEarlyInlining(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        cudaq::opt::createUnwindLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
    pm.addPass(createInlinerPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "Failure to synthesize callable block arguments in PyRemoteRESTQPU ");

    std::string moduleStr;
    {
      llvm::raw_string_ostream os(moduleStr);
      cloned.print(os);
    }
    // The remote rest qpu workflow will need the module string in
    // the internal registry.
    __cudaq_deviceCodeHolderAdd(kernelName.c_str(), moduleStr.c_str());
    return std::make_tuple(cloned, contextPtr.release());
  }

  void cleanupContext(MLIRContext *context) override { delete context; }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyRemoteRESTQPU, remote_rest)