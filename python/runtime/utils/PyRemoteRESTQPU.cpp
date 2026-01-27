/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteRESTQPU.h"
#include "common/RuntimeMLIR.h"

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

// We cannot use the RemoteRESTQPU since we'll get LLVM / MLIR statically loaded
// twice. We've extracted most of RemoteRESTQPU into BaseRemoteRESTQPU and will
// implement some core functionality here in PyRemoteRESTQPU so we don't load
// twice
class PyRemoteRESTQPU : public cudaq::BaseRemoteRESTQPU {
public:
  explicit PyRemoteRESTQPU() : BaseRemoteRESTQPU() { cudaq::initializeMLIR(); }

protected:
  std::tuple<ModuleOp, std::unique_ptr<MLIRContext>, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    auto [mod, ctx] = extractQuakeCodeAndContextImpl(kernelName);
    void *updatedArgs = nullptr;
    if (data) {
      auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(data);
      updatedArgs = wrapper->rawArgs;
    }
    return {mod, std::move(ctx), updatedArgs};
  }

  std::tuple<ModuleOp, std::unique_ptr<MLIRContext>>
  extractQuakeCodeAndContextImpl(const std::string &kernelName) {
    auto context = cudaq::getOwningMLIRContext();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, context.get());
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    // Here we have an opportunity to run any passes that are
    // specific to python before the rest of the RemoteRESTQPU workflow
    auto cloned = m_module->clone();
    PassManager pm(cloned.getContext());

    pm.addPass(cudaq::opt::createLambdaLifting());
    cudaq::opt::addAggressiveInlining(pm);
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(cudaq::opt::createApplySpecialization());
    pm.addPass(createInlinerPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
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
    return std::make_tuple(cloned, std::move(context));
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyRemoteRESTQPU, remote_rest)
