/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteRESTQPU.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;

extern "C" void __cudaq_deviceCodeHolderAdd(const char *, const char *);

namespace cudaq {

void registerToQIRTranslation();
void registerToOpenQASMTranslation();
void registerToIQMJsonTranslation();
void registerLLVMDialectTranslation(MLIRContext *context);

} // namespace cudaq

namespace cudaq {

class PyFermioniqRESTQPU : public cudaq::BaseRemoteRESTQPU {
protected:
  std::tuple<ModuleOp, MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {

    auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(data);
    auto m_module = wrapper->mod;
    auto callableNames = wrapper->callableNames;

    auto *context = m_module->getContext();
    static bool initOnce = [&] {
      registerToQIRTranslation();
      registerToOpenQASMTranslation();
      registerToIQMJsonTranslation();
      registerLLVMDialectTranslation(*context);
      return true;
    }();
    (void)initOnce;

    // Here we have an opportunity to run any passes that are
    // specific to python before the rest of the RemoteRESTQPU workflow
    auto cloned = m_module.clone();
    PassManager pm(cloned.getContext());
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(callableNames));
    cudaq::opt::addAggressiveEarlyInlining(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        cudaq::opt::createUnwindLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
    pm.addPass(createInlinerPass());
    pm.addPass(cudaq::opt::createExpandMeasurementsPass());
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
    return std::make_tuple(cloned, context, wrapper->rawArgs);
  }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyFermioniqRESTQPU, fermioniq)