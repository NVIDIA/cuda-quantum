/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/



#include "common/RuntimeMLIR.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/NoiseModel.h"
#include "cuda_runtime_api.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "cudaq.h"
#include <fstream>
#include <iostream>
#include <spdlog/cfg/env.h>

#include "JsonConvert.h"
#include "nvqir/CircuitSimulator.h"

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

namespace {

class RemoteSimulatorQPU : public cudaq::QPU {
private:
  std::string m_url;
  cudaq::RestClient m_client;
public:
  RemoteSimulatorQPU(const std::string &url, std::size_t id)
      : QPU(id), m_url(url) {}

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("Enqueue Task on QPU {}", qpu_id);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                                    std::uint64_t resultOffset) override {
    auto *sim = nvqir::getCircuitSimulatorInternal();
    cudaq::info("QPU::launchKernel named '{}' QPU {} (simulator = {})", name,
                qpu_id, sim->name());
    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(name);
    
    auto contextPtr = cudaq::initializeMLIR();
    MLIRContext &context = *contextPtr.get();

    auto module = parseSourceString<ModuleOp>(quakeCode, &context);
    if (!module)
      throw std::runtime_error("module cannot be parsed");

    // Extract the kernel name
    auto func = module->lookupSymbol<mlir::func::FuncOp>(
        std::string("__nvqpp__mlirgen__") + name);

    // Create a new Module to clone the function into
    auto location = FileLineColLoc::get(&context, "<builder>", 1, 1);
    ImplicitLocOpBuilder builder(location, &context);

    // FIXME this should be added to the builder.
    if (!func->hasAttr(cudaq::entryPointAttrName))
      func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
    auto moduleOp = builder.create<ModuleOp>();
    moduleOp.push_back(func.clone());

    if (args) {
      cudaq::info("Run Quake Synth.\n");
      PassManager pm(&context);
      pm.addPass(cudaq::opt::createQuakeSynthesizer(name, args));
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error("Could not successfully apply quake-synth.");
    }
    nlohmann::json job;
    job["kernel-name"] = name;
    job["quake"] = quakeCode;
    if (!executionContext)
      throw std::runtime_error("Invalid ExecutionContext encountered.");
    job["execution-context"] = *executionContext;
    std::map<std::string, std::string> headers {};
    auto resultJs = m_client.post(m_url, "job", job, headers);
    resultJs.get_to(*executionContext);
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::info("RemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    executionContext = context;
  }

  void resetExecutionContext() override {
    cudaq::info("RemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    // do nothing here
    executionContext = nullptr;
  }
};

class RemoteSimulatorQuantumPlatform : public cudaq::quantum_platform {
public:
  ~RemoteSimulatorQuantumPlatform() = default;
  RemoteSimulatorQuantumPlatform() {
    // Populate the information and add the QPUs
    platformQPUs.emplace_back(std::make_unique<RemoteSimulatorQPU>("localhost:3030/", 0));
    platformNumQPUs = platformQPUs.size();
  }

  bool supports_task_distribution() const override { return true; }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(RemoteSimulatorQuantumPlatform, remote_sim)
