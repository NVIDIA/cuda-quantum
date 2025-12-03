/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ArgumentConversion.h"
#include "common/Environment.h"
#include "common/JsonConvert.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "common/UnzipUtils.h"
#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
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
#include "mlir/Transforms/Passes.h"
#include <cxxabi.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
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

namespace cudaq {

class BaseRemoteRestRuntimeClient : public RemoteRuntimeClient {
protected:
  std::string m_url;
  static inline const std::vector<std::string> clientPasses = {};
  static inline const std::vector<std::string> serverPasses = {};
  // Random number generator.
  std::mt19937 randEngine{std::random_device{}()};

  static constexpr std::array<std::string_view, 1>
      DISALLOWED_EXECUTION_CONTEXT = {"tracer"};

  static constexpr bool isDisallowed(std::string_view context) {
    return std::any_of(DISALLOWED_EXECUTION_CONTEXT.begin(),
                       DISALLOWED_EXECUTION_CONTEXT.end(),
                       [context](std::string_view disallowed) {
                         return disallowed == context;
                       });
  }

  /// @brief Flag indicating whether we should enable MLIR printing before and
  /// after each pass. This is similar to `-mlir-print-ir-before-all` and
  /// `-mlir-print-ir-after-all` in `cudaq-opt`.
  bool enablePrintMLIREachPass = false;

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

  virtual mlir::ModuleOp
  lowerKernel(mlir::MLIRContext &mlirContext, const std::string &name,
              const void *args, std::uint64_t argsSize,
              const std::size_t startingArgIdx,
              const std::vector<void *> *rawArgs) override {
    enablePrintMLIREachPass = getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(name);
    auto module = parseSourceString<mlir::ModuleOp>(quakeCode, &mlirContext);
    if (!module)
      throw std::runtime_error("module cannot be parsed");

    // Extract the kernel name
    auto func = module->lookupSymbol<mlir::func::FuncOp>(
        std::string("__nvqpp__mlirgen__") + name);

    // Create a new Module to clone the function into
    auto location = mlir::FileLineColLoc::get(&mlirContext, "<builder>", 1, 1);
    mlir::ImplicitLocOpBuilder builder(location, &mlirContext);
    // Add CUDA-Q kernel attribute if not already set.
    if (!func->hasAttr(cudaq::kernelAttrName))
      func->setAttr(cudaq::kernelAttrName, builder.getUnitAttr());
    // Add entry-point attribute if not already set.
    if (!func->hasAttr(cudaq::entryPointAttrName))
      func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
    auto moduleOp = builder.create<mlir::ModuleOp>();
    moduleOp->setAttrs((*module)->getAttrDictionary());
    for (auto &op : *module) {
      if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
        // Add quantum kernels defined in the module.
        if (funcOp->hasAttr(cudaq::kernelAttrName) ||
            funcOp.getName().startswith("__nvqpp__mlirgen__") ||
            funcOp.getBody().empty())
          moduleOp.push_back(funcOp.clone());
      }
      // Add globals defined in the module.
      if (auto globalOp = dyn_cast<cc::GlobalOp>(op))
        moduleOp.push_back(globalOp.clone());
    }

    if (rawArgs || args) {
      mlir::PassManager pm(&mlirContext);
      if (rawArgs && !rawArgs->empty()) {
        CUDAQ_INFO("Run Argument Synth.\n");
        opt::ArgumentConverter argCon(name, moduleOp);
        argCon.gen_drop_front(*rawArgs, startingArgIdx);

        // Store kernel and substitution strings on the stack.
        // We pass string references to the `createArgumentSynthesisPass`.
        mlir::SmallVector<std::string> kernels;
        mlir::SmallVector<std::string> substs;
        for (auto *kInfo : argCon.getKernelSubstitutions()) {
          std::string kernName =
              cudaq::runtime::cudaqGenPrefixName + kInfo->getKernelName().str();
          kernels.emplace_back(kernName);
          std::string substBuff;
          llvm::raw_string_ostream ss(substBuff);
          ss << kInfo->getSubstitutionModule();
          substs.emplace_back(substBuff);
        }

        // Collect references for the argument synthesis.
        mlir::SmallVector<mlir::StringRef> kernelRefs{kernels.begin(),
                                                      kernels.end()};
        mlir::SmallVector<mlir::StringRef> substRefs{substs.begin(),
                                                     substs.end()};
        pm.addPass(opt::createArgumentSynthesisPass(kernelRefs, substRefs));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(opt::createDeleteStates());
        pm.addNestedPass<mlir::func::FuncOp>(
            opt::createReplaceStateWithKernel());
        pm.addPass(mlir::createSymbolDCEPass());
      } else if (args) {
        CUDAQ_INFO("Run Quake Synth.\n");
        pm.addPass(opt::createQuakeSynthesizer(name, args, startingArgIdx));
      }
      pm.addPass(mlir::createCanonicalizerPass());
      if (enablePrintMLIREachPass) {
        moduleOp.getContext()->disableMultithreading();
        pm.enableIRPrinting();
      }
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error("Could not successfully apply quake-synth.");
    }

    // Note: do not run state preparation pass here since we are always
    // using simulators.

    // Run client-side passes. `clientPasses` is empty right now, but the code
    // below accommodates putting passes into it.
    mlir::PassManager pm(&mlirContext);
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);

    std::string pipeline =
        std::accumulate(clientPasses.begin(), clientPasses.end(), std::string(),
                        [](const auto &ss, const auto &s) {
                          return ss.empty() ? s : ss + "," + s;
                        });
    // TODO: replace environment variable with runtime configuration
    if (getEnvBool("CUDAQ_PHASE_FOLDING", true)) {
      if (getEnvBool("CUDAQ_BYPASS_PHASE_FOLDING_MINS", false))
        pipeline =
            pipeline + "phase-folding-pipeline{min-length=0 min-rz-weight=0}";
      else
        pipeline = pipeline + "phase-folding-pipeline";
    }

    if (enablePrintMLIREachPass) {
      moduleOp.getContext()->disableMultithreading();
      pm.enableIRPrinting();
    }

    if (failed(parsePassPipeline(pipeline, pm, os)))
      throw std::runtime_error(
          "Remote rest platform failed to add passes to pipeline (" + errMsg +
          ").");

    mlir::DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run
    if (failed(pm.run(moduleOp)))
      throw std::runtime_error(
          "Remote rest platform: applying IR passes failed.");

    return moduleOp;
  }

  std::string constructKernelPayload(mlir::MLIRContext &mlirContext,
                                     const std::string &name, const void *args,
                                     std::uint64_t voidStarSize,
                                     std::size_t startingArgIdx,
                                     const std::vector<void *> *rawArgs) {
    ScopedTraceWithContext(cudaq::TIMING_JIT, "constructKernelPayload");
    auto moduleOp = lowerKernel(mlirContext, name, args, voidStarSize,
                                startingArgIdx, rawArgs);

    mlir::PassManager pm(&mlirContext);
    // For now, the server side expects full-QIR.
    opt::addAOTPipelineConvertToQIR(pm);

    if (failed(pm.run(moduleOp)))
      throw std::runtime_error(
          "Remote rest platform: applying IR passes failed.");

    std::string mlirCode;
    llvm::raw_string_ostream outStr(mlirCode);
    mlir::OpPrintingFlags opf;
    opf.enableDebugInfo(/*enable=*/true,
                        /*pretty=*/false);
    moduleOp.print(outStr, opf);
    return llvm::encodeBase64(mlirCode);
  }

  cudaq::RestRequest constructVQEJobRequest(
      mlir::MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
      const std::string &backendSimName, const std::string &kernelName,
      const void *kernelArgs, cudaq::gradient *gradient,
      cudaq::optimizer &optimizer, const int n_params,
      const std::vector<void *> *rawArgs) {
    cudaq::RestRequest request(io_context, version());

    request.opt = RestRequestOptFields();
    request.opt->optimizer_n_params = n_params;
    request.opt->optimizer_type = get_optimizer_type(optimizer);
    request.opt->optimizer_ptr = &optimizer;
    request.opt->gradient_ptr = gradient;
    if (gradient)
      request.opt->gradient_type = get_gradient_type(*gradient);

    request.entryPoint = kernelName;
    request.passes = serverPasses;
    request.format = cudaq::CodeFormat::MLIR;
    request.code =
        constructKernelPayload(mlirContext, kernelName,
                               /*kernelArgs=*/kernelArgs,
                               /*argsSize=*/0, /*startingArgIdx=*/1, rawArgs);
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

  cudaq::RestRequest constructJobRequest(
      mlir::MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
      const std::string &backendSimName, const std::string &kernelName,
      void (*kernelFunc)(void *), const void *kernelArgs,
      std::uint64_t argsSize, const std::vector<void *> *rawArgs) {

    cudaq::RestRequest request(io_context, version());
    request.entryPoint = kernelName;
    request.passes = serverPasses;
    request.format = cudaq::CodeFormat::MLIR;

    if (io_context.name == "state-overlap") {
      if (!io_context.overlapComputeStates.has_value())
        throw std::runtime_error("Invalid execution context: no input states");
      const auto *castedState1 = dynamic_cast<const RemoteSimulationState *>(
          io_context.overlapComputeStates->first);
      const auto *castedState2 = dynamic_cast<const RemoteSimulationState *>(
          io_context.overlapComputeStates->second);
      if (!castedState1 || !castedState2)
        throw std::runtime_error(
            "Invalid execution context: input states are not compatible");
      if (!castedState1->getKernelInfo().has_value())
        throw std::runtime_error("Missing first input state in state-overlap");
      if (!castedState2->getKernelInfo().has_value())
        throw std::runtime_error("Missing second input state in state-overlap");
      auto [kernelName1, args1] = castedState1->getKernelInfo().value();
      auto [kernelName2, args2] = castedState2->getKernelInfo().value();
      cudaq::IRPayLoad stateIrPayload1, stateIrPayload2;

      stateIrPayload1.entryPoint = kernelName1;
      stateIrPayload1.ir =
          constructKernelPayload(mlirContext, kernelName1, nullptr, 0,
                                 /*startingArgIdx=*/0, &args1);
      stateIrPayload2.entryPoint = kernelName2;
      stateIrPayload2.ir =
          constructKernelPayload(mlirContext, kernelName2, nullptr, 0,
                                 /*startingArgIdx=*/0, &args2);
      // First kernel of the overlap calculation
      request.code = stateIrPayload1.ir;
      request.entryPoint = stateIrPayload1.entryPoint;
      // Second kernel of the overlap calculation
      request.overlapKernel = stateIrPayload2;
    } else {
      request.code =
          constructKernelPayload(mlirContext, kernelName, kernelArgs, argsSize,
                                 /*startingArgIdx=*/0, rawArgs);
    }
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
  sendRequest(mlir::MLIRContext &mlirContext,
              cudaq::ExecutionContext &io_context,
              cudaq::gradient *vqe_gradient, cudaq::optimizer *vqe_optimizer,
              const int vqe_n_params, const std::string &backendSimName,
              const std::string &kernelName, void (*kernelFunc)(void *),
              const void *kernelArgs, std::uint64_t argsSize,
              std::string *optionalErrorMsg,
              const std::vector<void *> *rawArgs) override {
    if (isDisallowed(io_context.name))
      throw std::runtime_error(
          io_context.name +
          " operation is not supported with cudaq target remote-mqpu!");

    cudaq::RestRequest request = [&]() {
      if (vqe_n_params > 0)
        return constructVQEJobRequest(mlirContext, io_context, backendSimName,
                                      kernelName, kernelArgs, vqe_gradient,
                                      *vqe_optimizer, vqe_n_params, rawArgs);
      return constructJobRequest(mlirContext, io_context, backendSimName,
                                 kernelName, kernelFunc, kernelArgs, argsSize,
                                 rawArgs);
    }();

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
      CUDAQ_DBG("Response: {}", resultJs.dump(/*indent=*/2));

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
    } catch (...) {
      std::string exType = __cxxabiv1::__cxa_current_exception_type()->name();
      auto demangledPtr =
          __cxxabiv1::__cxa_demangle(exType.c_str(), nullptr, nullptr, nullptr);
      if (demangledPtr && optionalErrorMsg) {
        std::string demangledName(demangledPtr);
        *optionalErrorMsg = "Unhandled exception of type " + demangledName;
      } else if (optionalErrorMsg) {
        *optionalErrorMsg = "Unhandled exception of unknown type";
      }
      return false;
    }
  }

  virtual void resetRemoteRandomSeed(std::size_t seed) override {
    // Re-seed the generator, e.g., when `cudaq::set_random_seed` is called.
    randEngine.seed(seed);
  }

  // The remote-mqpu backend (this class) returns true for all remote
  // capabilities unless overridden by environment variable.
  virtual RemoteCapabilities getRemoteCapabilities() const override {
    // Default to all true, but allow the user to override to all false.
    if (getEnvBool("CUDAQ_CLIENT_REMOTE_CAPABILITY_OVERRIDE", true))
      return RemoteCapabilities(/*initValues=*/true);
    return RemoteCapabilities(/*initValues=*/false);
  }
};

} // namespace cudaq
