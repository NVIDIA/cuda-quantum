/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
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

#include "common/JsonConvert.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RestClient.h"
#include "common/UnzipUtils.h"
#include "cudaq.h"

#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <streambuf>

namespace {
/// Util class to execute a functor when an object of this class goes
/// out-of-scope.
// This can be used to perform some clean up.
// ```
// {
//   ScopeExit cleanUp(f);
//   ...
// } <- f() is called to perform some cleanup action.
// ```
struct ScopeExit {
  ScopeExit(std::function<void()> &&func) : m_atExitFunc(std::move(func)) {}
  ~ScopeExit() noexcept { m_atExitFunc(); }
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
  ScopeExit(ScopeExit &&other) = delete;
  ScopeExit &operator=(ScopeExit &&other) = delete;

private:
  std::function<void()> m_atExitFunc;
};
} // namespace

using namespace mlir;

namespace cudaq {
class BaseRemoteRestRuntimeClient : public cudaq::RemoteRuntimeClient {
protected:
  std::string m_url;
  static inline const std::vector<std::string> clientPasses = {
      "func.func(unwind-lowering)",
      "func.func(indirect-to-direct-calls)",
      "inline",
      "canonicalize",
      "apply-op-specialization",
      "func.func(apply-control-negations)",
      "func.func(memtoreg{quantum=0})",
      "canonicalize",
      "expand-measurements",
      "cc-loop-normalize",
      "cc-loop-unroll",
      "canonicalize",
      "func.func(add-dealloc)",
      "func.func(quake-add-metadata)",
      "canonicalize",
      "func.func(lower-to-cfg)",
      "func.func(combine-quantum-alloc)",
      "canonicalize",
      "cse",
      "quake-to-qir"};
  static inline const std::vector<std::string> serverPasses = {};
  // Random number generator.
  std::mt19937 randEngine{std::random_device{}()};

public:
  virtual void setConfig(
      const std::unordered_map<std::string, std::string> &configs) override {
    const auto urlIter = configs.find("url");
    if (urlIter != configs.end())
      m_url = urlIter->second;
  }

  virtual int version() const override {
    // Check if we have an environment variable override
    if (auto *envVal = std::getenv("CUDAQ_REST_CLIENT_VERSION"))
      return std::stoi(envVal);

    // Otherwise, just use the version defined in the code.
    return cudaq::RestRequest::REST_PAYLOAD_VERSION;
  }

  std::string constructKernelPayload(MLIRContext &mlirContext,
                                     const std::string &name,
                                     void (*kernelFunc)(void *), void *args,
                                     std::uint64_t voidStarSize) {
    if (cudaq::__internal__::isLibraryMode(name)) {
      // Library mode: retrieve the embedded bitcode in the executable.
      const auto path = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
      // Load the object file
      auto [objBin, objBuffer] =
          llvm::cantFail(llvm::object::ObjectFile::createObjectFile(path))
              .takeBinary();
      if (!objBin)
        throw std::runtime_error("Failed to load binary object file");
      for (const auto &section : objBin->sections()) {
        // Get the bitcode section
        if (section.isBitcode()) {
          llvm::MemoryBufferRef llvmBc(llvm::cantFail(section.getContents()),
                                       "Bitcode");
          return llvm::encodeBase64(llvmBc.getBuffer());
        }
      }
      return "";
    } else {
      // Get the quake representation of the kernel
      auto quakeCode = cudaq::get_quake_by_name(name);
      auto module = parseSourceString<ModuleOp>(quakeCode, &mlirContext);
      if (!module)
        throw std::runtime_error("module cannot be parsed");

      // Extract the kernel name
      auto func = module->lookupSymbol<mlir::func::FuncOp>(
          std::string("__nvqpp__mlirgen__") + name);

      // Create a new Module to clone the function into
      auto location = FileLineColLoc::get(&mlirContext, "<builder>", 1, 1);
      ImplicitLocOpBuilder builder(location, &mlirContext);
      // Add cuda quantum kernel attribute if not already set.
      if (!func->hasAttr(cudaq::kernelAttrName))
        func->setAttr(cudaq::kernelAttrName, builder.getUnitAttr());
      // Add entry-point attribute if not already set.
      if (!func->hasAttr(cudaq::entryPointAttrName))
        func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
      auto moduleOp = builder.create<ModuleOp>();
      for (auto &op : *module) {
        auto funcOp = dyn_cast<func::FuncOp>(op);
        // Add quantum kernels defined in the module.
        if (funcOp && (funcOp->hasAttr(cudaq::kernelAttrName) ||
                       funcOp.getName().startswith("__nvqpp__mlirgen__")))
          moduleOp.push_back(funcOp.clone());
      }

      if (args) {
        cudaq::info("Run Quake Synth.\n");
        PassManager pm(&mlirContext);
        pm.addPass(cudaq::opt::createQuakeSynthesizer(name, args));
        if (failed(pm.run(moduleOp)))
          throw std::runtime_error("Could not successfully apply quake-synth.");
      }

      // Client-side passes
      if (!clientPasses.empty()) {
        PassManager pm(&mlirContext);
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        const std::string pipeline =
            std::accumulate(clientPasses.begin(), clientPasses.end(),
                            std::string(), [](const auto &ss, const auto &s) {
                              return ss.empty() ? s : ss + "," + s;
                            });
        if (failed(parsePassPipeline(pipeline, pm, os)))
          throw std::runtime_error(
              "Remote rest platform failed to add passes to pipeline (" +
              errMsg + ").");

        if (failed(pm.run(moduleOp)))
          throw std::runtime_error(
              "Remote rest platform: applying IR passes failed.");
      }
      std::string mlirCode;
      llvm::raw_string_ostream outStr(mlirCode);
      mlir::OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      moduleOp.print(outStr, opf);
      return llvm::encodeBase64(mlirCode);
    }
  }

  cudaq::RestRequest constructJobRequest(
      MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
      const std::string &backendSimName, const std::string &kernelName,
      void (*kernelFunc)(void *), void *kernelArgs, std::uint64_t argsSize) {

    cudaq::RestRequest request(io_context, version());
    request.entryPoint = kernelName;
    if (cudaq::__internal__::isLibraryMode(kernelName)) {
      request.format = cudaq::CodeFormat::LLVM;
      if (kernelArgs && argsSize > 0) {
        cudaq::info("Serialize {} bytes of args.", argsSize);
        request.args.resize(argsSize);
        std::memcpy(request.args.data(), kernelArgs, argsSize);
      }

      if (kernelFunc) {
        ::Dl_info info;
        ::dladdr(reinterpret_cast<void *>(kernelFunc), &info);
        const auto funcName = cudaq::quantum_platform::demangle(info.dli_sname);
        cudaq::info("RemoteSimulatorQPU: retrieve name '{}' for kernel {}",
                    funcName, kernelName);
        request.entryPoint = funcName;
      }
    } else {
      request.passes = serverPasses;
      request.format = cudaq::CodeFormat::MLIR;
    }

    request.code = constructKernelPayload(mlirContext, kernelName, kernelFunc,
                                          kernelArgs, argsSize);
    request.simulator = backendSimName;
    // Remote server seed
    // Note: unlike local executions whereby a static instance of the simulator
    // is seeded once when `cudaq::set_random_seed` is called, thus not being
    // re-seeded between executions. For remote executions, we use the runtime
    // level seed value to seed a random number generator to seed the server.
    // i.e., consecutive remote executions on the server from the same client
    // session (where `cudaq::set_random_seed` is called), get new random seeds
    // for each execution. The sequence is still deterministic based on the
    // runtime-level seed value.
    request.seed = [&]() {
      std::uniform_int_distribution<std::size_t> seedGen(
          std::numeric_limits<std::size_t>::min(),
          std::numeric_limits<std::size_t>::max());
      return seedGen(randEngine);
    }();
    return request;
  }

  virtual bool
  sendRequest(MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
              const std::string &backendSimName, const std::string &kernelName,
              void (*kernelFunc)(void *), void *kernelArgs,
              std::uint64_t argsSize, std::string *optionalErrorMsg) override {
    cudaq::RestRequest request =
        constructJobRequest(mlirContext, io_context, backendSimName, kernelName,
                            kernelFunc, kernelArgs, argsSize);
    if (request.code.empty()) {
      if (optionalErrorMsg)
        *optionalErrorMsg =
            std::string(
                "Failed to construct/retrieve kernel IR for kernel named ") +
            kernelName;
      return false;
    }

    // Don't let curl adding "Expect: 100-continue" header, which is not
    // suitable for large requests, e.g., bitcode in the JSON request.
    //  Ref: https://gms.tf/when-curl-sends-100-continue.html
    std::map<std::string, std::string> headers{
        {"Expect:", ""}, {"Content-type", "application/json"}};
    json requestJson = request;
    try {
      cudaq::RestClient restClient;
      auto resultJs =
          restClient.post(m_url, "job", requestJson, headers, false);

      if (!resultJs.contains("executionContext")) {
        std::stringstream errorMsg;
        if (resultJs.contains("status")) {
          errorMsg << "Failed to execute the kernel on the remote server: "
                   << resultJs["status"] << "\n";
          if (resultJs.contains("errorMessage")) {
            errorMsg << "Error message: " << resultJs["errorMessage"] << "\n";
          }
        } else {
          errorMsg << "Failed to execute the kernel on the remote server.\n";
          errorMsg << "Unexpected response from the REST server. Missing the "
                      "required field 'executionContext'.";
        }
        if (optionalErrorMsg)
          *optionalErrorMsg = errorMsg.str();
        return false;
      }
      resultJs["executionContext"].get_to(io_context);
      return true;
    } catch (std::exception &e) {
      if (optionalErrorMsg)
        *optionalErrorMsg = e.what();
      return false;
    }
  }

  virtual void resetRemoteRandomSeed(std::size_t seed) override {
    // Re-seed the generator, e.g., when `cudaq::set_random_seed` is called.
    randEngine.seed(seed);
  }
};
} // namespace cudaq
