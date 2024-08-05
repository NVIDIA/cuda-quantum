/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/spin_op.h"
#include "nvqpp_config.h"
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
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

namespace cudaq {

class BaseRemoteRESTQPU : public cudaq::QPU {
protected:
  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path
  std::filesystem::path platformPath;

  /// @brief The Pass pipeline string, configured by the
  /// QPU configuration file in the platform path.
  std::string passPipelineConfig = "canonicalize";

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Name of code generation target (e.g. `qir-adaptive`, `qir-base`,
  /// `qasm2`, `iqm`)
  std::string codegenTranslation = "";

  /// @brief Additional passes to run after the codegen-specific passes
  std::string postCodeGenPasses = "";

  // Pointer to the concrete Executor for this QPU
  std::unique_ptr<cudaq::Executor> executor;

  /// @brief Pointer to the concrete ServerHelper, provides
  /// specific JSON payloads and POST/GET URL paths.
  std::unique_ptr<cudaq::ServerHelper> serverHelper;

  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Flag indicating whether we should emulate
  /// execution locally.
  bool emulate = false;

  /// @brief Flag indicating whether we should print the IR.
  bool printIR = false;

  /// @brief Flag indicating whether we should perform the passes in a
  /// single-threaded environment, useful for debug. Similar to
  /// `-mlir-disable-threading` for `cudaq-opt`.
  bool disableMLIRthreading = false;

  /// @brief Flag indicating whether we should enable MLIR printing before and
  /// after each pass. This is similar to `-mlir-print-ir-before-all` and
  /// `-mlir-print-ir-after-all` in `cudaq-opt`.
  bool enablePrintMLIREachPass = false;

  /// @brief If we are emulating locally, keep track
  /// of JIT engines for invoking the kernels.
  std::vector<mlir::ExecutionEngine *> jitEngines;

  /// @brief Invoke the kernel in the JIT engine
  void invokeJITKernel(mlir::ExecutionEngine *jit,
                       const std::string &kernelName) {
    auto funcPtr = jit->lookup(std::string("__nvqpp__mlirgen__") + kernelName);
    if (!funcPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get kernelReg function.");
    }
    reinterpret_cast<void (*)()>(*funcPtr)();
  }

  /// @brief Invoke the kernel in the JIT engine and then delete the JIT engine.
  void invokeJITKernelAndRelease(mlir::ExecutionEngine *jit,
                                 const std::string &kernelName) {
    invokeJITKernel(jit, kernelName);
    delete jit;
  }

  virtual std::tuple<mlir::ModuleOp, mlir::MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName, void *data) = 0;
  virtual void cleanupContext(mlir::MLIRContext *context) { return; }

public:
  /// @brief The constructor
  BaseRemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<cudaq::Executor>();
  }

  BaseRemoteRESTQPU(BaseRemoteRESTQPU &&) = delete;
  virtual ~BaseRemoteRESTQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  /// @return
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supportsConditionalFeedback() override {
    return codegenTranslation == "qir-adaptive";
  }

  /// Provide the number of shots
  void setShots(int _nShots) override {
    nShots = _nShots;
    executor->setShots(static_cast<std::size_t>(_nShots));
  }

  /// Clear the number of shots
  void clearShots() override { nShots = std::nullopt; }
  virtual bool isRemote() override { return !emulate; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return emulate; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  void setNoiseModel(const cudaq::noise_model *model) override {
    if (!emulate && model)
      throw std::runtime_error(
          "Noise modeling is not allowed on remote physical quantum backends.");

    noiseModel = model;
  }

  /// Store the execution context for launchKernel
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    if (!context)
      return;

    cudaq::info("Remote Rest QPU setting execution context to {}",
                context->name);

    // Execution context is valid
    executionContext = context;
  }

  /// Reset the execution context
  void resetExecutionContext() override {
    // do nothing here
    executionContext = nullptr;
  }

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file (bundled as part of this
  /// CUDA-Q installation) and extract MLIR lowering pipelines and
  /// specific code generation output required by this backend (QIR/QASM2).
  void setTargetBackend(const std::string &backend) override {
    cudaq::info("Remote REST platform is targeting {}.", backend);

    // First we see if the given backend has extra config params
    auto mutableBackend = backend;
    if (mutableBackend.find(";") != std::string::npos) {
      auto split = cudaq::split(mutableBackend, ';');
      mutableBackend = split[0];
      // Must be key-value pairs, therefore an even number of values here
      if ((split.size() - 1) % 2 != 0)
        throw std::runtime_error(
            "Backend config must be provided as key-value pairs: " +
            std::to_string(split.size()));

      // Add to the backend configuration map
      for (std::size_t i = 1; i < split.size(); i += 2) {
        // No need to decode trivial true/false values
        if (split[i + 1].starts_with("base64_")) {
          split[i + 1].erase(0, 7); // erase "base64_"
          std::vector<char> decoded_vec;
          if (auto err = llvm::decodeBase64(split[i + 1], decoded_vec))
            throw std::runtime_error("DecodeBase64 error");
          std::string decodedStr(decoded_vec.data(), decoded_vec.size());
          cudaq::info("Decoded {} parameter from '{}' to '{}'", split[i],
                      split[i + 1], decodedStr);
          backendConfig.insert({split[i], decodedStr});
        } else {
          backendConfig.insert({split[i], split[i + 1]});
        }
      }
    }

    // Turn on emulation mode if requested
    auto iter = backendConfig.find("emulate");
    emulate = iter != backendConfig.end() && iter->second == "true";

    // Print the IR if requested
    printIR = getEnvBool("CUDAQ_DUMP_JIT_IR", printIR);

    // Get additional debug values
    disableMLIRthreading =
        getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", disableMLIRthreading);
    enablePrintMLIREachPass =
        getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", enablePrintMLIREachPass);

    // If the very verbose enablePrintMLIREachPass flag is set, then
    // multi-threading must be disabled.
    if (enablePrintMLIREachPass) {
      disableMLIRthreading = true;
    }

    /// Once we know the backend, we should search for the configuration file
    /// from there we can get the URL/PORT and the required MLIR pass
    /// pipeline.
    std::string fileName = mutableBackend + std::string(".yml");
    auto configFilePath = platformPath / fileName;
    cudaq::info("Config file path = {}", configFilePath.string());
    std::ifstream configFile(configFilePath.string());
    std::string configYmlContents((std::istreambuf_iterator<char>(configFile)),
                                  std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configYmlContents.c_str());
    Input >> config;
    if (config.BackendConfig.has_value()) {
      if (!config.BackendConfig->PlatformLoweringConfig.empty()) {
        cudaq::info("Appending lowering pipeline: {}",
                    config.BackendConfig->PlatformLoweringConfig);
        passPipelineConfig +=
            "," + config.BackendConfig->PlatformLoweringConfig;
      }
      if (!config.BackendConfig->CodegenEmission.empty()) {
        cudaq::info("Set codegen translation: {}",
                    config.BackendConfig->CodegenEmission);
        codegenTranslation = config.BackendConfig->CodegenEmission;
      }
      if (!config.BackendConfig->PostCodeGenPasses.empty()) {
        cudaq::info("Adding post-codegen lowering pipeline: {}",
                    config.BackendConfig->PostCodeGenPasses);
        postCodeGenPasses = config.BackendConfig->PostCodeGenPasses;
      }
    }
    std::string allowEarlyExitSetting =
        (codegenTranslation == "qir-adaptive") ? "1" : "0";
    passPipelineConfig = std::string("cc-loop-unroll{allow-early-exit=") +
                         allowEarlyExitSetting + "}," + passPipelineConfig;

    auto disableQM = backendConfig.find("disable_qubit_mapping");
    if (disableQM != backendConfig.end() && disableQM->second == "true") {
      // Replace the qubit-mapping{device=<>} with
      // qubit-mapping{device=bypass} to effectively disable the qubit-mapping
      // pass. Use $1 - $4 to make sure any other pass options are left
      // untouched.
      std::regex qubitMapping(
          "(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
      std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
      passPipelineConfig =
          std::regex_replace(passPipelineConfig, qubitMapping, replacement);
      cudaq::info("disable_qubit_mapping option found, so updated lowering "
                  "pipeline to {}",
                  passPipelineConfig);
    }

    // Set the qpu name
    qpuName = mutableBackend;

    // Create the ServerHelper for this QPU and give it the backend config
    serverHelper = cudaq::registry::get<cudaq::ServerHelper>(qpuName);
    serverHelper->initialize(backendConfig);
    serverHelper->updatePassPipeline(platformPath, passPipelineConfig);

    // Give the server helper to the executor
    executor->setServerHelper(serverHelper.get());
  }

  /// @brief Conditionally form an output_names JSON object if this was for QIR
  nlohmann::json formOutputNames(const std::string &codegenTranslation,
                                 const std::string &codeStr) {
    // Form an output_names mapping from codeStr
    nlohmann::json output_names;
    std::vector<char> bitcode;
    if (codegenTranslation.starts_with("qir")) {
      // decodeBase64 will throw a runtime exception if it fails
      if (llvm::decodeBase64(codeStr, bitcode)) {
        cudaq::info("Could not decode codeStr {}", codeStr);
      } else {
        llvm::LLVMContext llvmContext;
        auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
            llvm::StringRef(bitcode.data(), bitcode.size()));
        auto moduleOrError =
            llvm::parseBitcodeFile(buffer->getMemBufferRef(), llvmContext);
        if (moduleOrError.takeError())
          throw std::runtime_error("Could not parse bitcode file");
        auto module = std::move(moduleOrError.get());
        for (llvm::Function &func : *module) {
          if (func.hasFnAttribute("entry_point") &&
              func.hasFnAttribute("output_names")) {
            output_names = nlohmann::json::parse(
                func.getFnAttribute("output_names").getValueAsString());
            break;
          }
        }
      }
    }
    return output_names;
  }

  /// @brief Extract the Quake representation for the given kernel name and
  /// lower it to the code format required for the specific backend. The
  /// lowering process is controllable via the configuration file in the
  /// platform directory for the targeted backend.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs) {

    auto [m_module, contextPtr, updatedArgs] =
        extractQuakeCodeAndContext(kernelName, kernelArgs);

    mlir::MLIRContext &context = *contextPtr;

    // Extract the kernel name
    auto func = m_module.lookupSymbol<mlir::func::FuncOp>(
        std::string("__nvqpp__mlirgen__") + kernelName);

    // Create a new Module to clone the function into
    auto location = mlir::FileLineColLoc::get(&context, "<builder>", 1, 1);
    mlir::ImplicitLocOpBuilder builder(location, &context);

    // FIXME this should be added to the builder.
    if (!func->hasAttr(cudaq::entryPointAttrName))
      func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
    auto moduleOp = builder.create<mlir::ModuleOp>();
    moduleOp.push_back(func.clone());
    moduleOp->setAttrs(m_module->getAttrDictionary());

    for (auto &op : m_module.getOps()) {
      // Add any global symbols, including global constant arrays.
      // Global constant arrays can be created during compilation,
      // `lift-array-value`, `quake-synthesizer`, and `get-concrete-matrix`
      // passes.
      if (auto globalOp = dyn_cast<cudaq::cc::GlobalOp>(op))
        moduleOp.push_back(globalOp.clone());
    }

    // Lambda to apply a specific pipeline to the given ModuleOp
    auto runPassPipeline = [&](const std::string &pipeline,
                               mlir::ModuleOp moduleOpIn) {
      mlir::PassManager pm(&context);
      std::string errMsg;
      llvm::raw_string_ostream os(errMsg);
      cudaq::info("Pass pipeline for {} = {}", kernelName, pipeline);
      if (failed(parsePassPipeline(pipeline, pm, os)))
        throw std::runtime_error(
            "Remote rest platform failed to add passes to pipeline (" + errMsg +
            ").");
      if (disableMLIRthreading || enablePrintMLIREachPass)
        moduleOpIn.getContext()->disableMultithreading();
      if (enablePrintMLIREachPass)
        pm.enableIRPrinting();
      if (failed(pm.run(moduleOpIn)))
        throw std::runtime_error("Remote rest platform Quake lowering failed.");
    };

    if (updatedArgs) {
      cudaq::info("Run Quake Synth.\n");
      mlir::PassManager pm(&context);
      pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, updatedArgs));
      pm.addPass(mlir::createCanonicalizerPass());
      if (disableMLIRthreading || enablePrintMLIREachPass)
        moduleOp.getContext()->disableMultithreading();
      if (enablePrintMLIREachPass)
        pm.enableIRPrinting();
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error("Could not successfully apply quake-synth.");
    }

    runPassPipeline(passPipelineConfig, moduleOp);

    auto entryPointFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(
        std::string("__nvqpp__mlirgen__") + kernelName);
    std::vector<std::size_t> mapping_reorder_idx;
    if (auto mappingAttr = dyn_cast_if_present<mlir::ArrayAttr>(
            entryPointFunc->getAttr("mapping_reorder_idx"))) {
      mapping_reorder_idx.resize(mappingAttr.size());
      std::transform(mappingAttr.begin(), mappingAttr.end(),
                     mapping_reorder_idx.begin(), [](mlir::Attribute attr) {
                       return mlir::cast<mlir::IntegerAttr>(attr).getInt();
                     });
    }

    if (executionContext) {
      if (executionContext->name == "sample")
        executionContext->reorderIdx = mapping_reorder_idx;
      else
        executionContext->reorderIdx.clear();
    }

    std::vector<std::pair<std::string, mlir::ModuleOp>> modules;
    // Apply observations if necessary
    if (executionContext && executionContext->name == "observe") {
      mapping_reorder_idx.clear();
      runPassPipeline("canonicalize,cse", moduleOp);
      cudaq::spin_op &spin = *executionContext->spin.value();
      for (const auto &term : spin) {
        if (term.is_identity())
          continue;

        // Get the ansatz
        auto ansatz = moduleOp.lookupSymbol<mlir::func::FuncOp>(
            std::string("__nvqpp__mlirgen__") + kernelName);

        // Create a new Module to clone the ansatz into it
        auto tmpModuleOp = builder.create<mlir::ModuleOp>();
        tmpModuleOp.push_back(ansatz.clone());

        // Extract the binary symplectic encoding
        auto [binarySymplecticForm, coeffs] = term.get_raw_data();

        // Create the pass manager, add the quake observe ansatz pass
        // and run it followed by the canonicalizer
        mlir::PassManager pm(&context);
        pm.addNestedPass<mlir::func::FuncOp>(
            cudaq::opt::createObserveAnsatzPass(binarySymplecticForm[0]));
        if (disableMLIRthreading || enablePrintMLIREachPass)
          tmpModuleOp.getContext()->disableMultithreading();
        if (enablePrintMLIREachPass)
          pm.enableIRPrinting();
        if (failed(pm.run(tmpModuleOp)))
          throw std::runtime_error("Could not apply measurements to ansatz.");
        runPassPipeline(passPipelineConfig, tmpModuleOp);
        modules.emplace_back(term.to_string(false), tmpModuleOp);
      }
    } else
      modules.emplace_back(kernelName, moduleOp);

    if (emulate) {
      // If we are in emulation mode, we need to first get a
      // full QIR representation of the code. Then we'll map to
      // an LLVM Module, create a JIT ExecutionEngine pointer
      // and use that for execution
      for (auto &[name, module] : modules) {
        auto clonedModule = module.clone();
        jitEngines.emplace_back(
            cudaq::createQIRJITEngine(clonedModule, codegenTranslation));
      }
    }

    // Get the code gen translation
    auto translation = cudaq::getTranslation(codegenTranslation);

    // Apply user-specified codegen
    std::vector<cudaq::KernelExecution> codes;
    for (auto &[name, moduleOpI] : modules) {
      std::string codeStr;
      {
        llvm::raw_string_ostream outStr(codeStr);
        if (disableMLIRthreading)
          moduleOpI.getContext()->disableMultithreading();
        if (failed(translation(moduleOpI, outStr, postCodeGenPasses, printIR,
                               enablePrintMLIREachPass)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      }

      // Form an output_names mapping from codeStr
      nlohmann::json j = formOutputNames(codegenTranslation, codeStr);

      codes.emplace_back(name, codeStr, j, mapping_reorder_idx);
    }

    cleanupContext(contextPtr);
    return codes;
  }

  /// @brief Launch the kernel. Extract the Quake code and lower to
  /// the representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as asynchronous or
  /// synchronous invocation.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::draw().");

    // Get the Quake code, lowered according to config file.
    auto codes = lowerQuakeCode(kernelName, args);

    // After performing lowerQuakeCode, check to see if we are simply drawing
    // the circuit. If so, perform the trace here and then return.
    if (executionContext->name == "tracer" && jitEngines.size() == 1) {
      cudaq::getExecutionManager()->setExecutionContext(executionContext);
      invokeJITKernelAndRelease(jitEngines[0], kernelName);
      cudaq::getExecutionManager()->resetExecutionContext();
      jitEngines.clear();
      return;
    }

    // Get the current execution context and number of shots
    std::size_t localShots = 1000;
    if (executionContext->shots != std::numeric_limits<std::size_t>::max() &&
        executionContext->shots != 0)
      localShots = executionContext->shots;

    executor->setShots(localShots);

    // If emulation requested, then just grab the function
    // and invoke it with the simulator
    cudaq::details::future future;
    if (emulate) {

      // Fetch the thread-specific seed outside and then pass it inside.
      std::size_t seed = cudaq::get_random_seed();

      // Launch the execution of the simulated jobs asynchronously
      future = cudaq::details::future(std::async(
          std::launch::async,
          [&, codes, localShots, kernelName, seed,
           reorderIdx = executionContext->reorderIdx,
           localJIT = std::move(jitEngines)]() mutable -> cudaq::sample_result {
            std::vector<cudaq::ExecutionResult> results;

            // If seed is 0, then it has not been set.
            if (seed > 0)
              cudaq::set_random_seed(seed);

            bool hasConditionals =
                cudaq::kernelHasConditionalFeedback(kernelName);
            if (hasConditionals && codes.size() > 1)
              throw std::runtime_error("error: spin_ops not yet supported with "
                                       "kernels containing conditionals");
            if (hasConditionals) {
              executor->setShots(1); // run one shot at a time

              // If this is adaptive profile and the kernel has conditionals,
              // then you have to run the code localShots times instead of
              // running the kernel once and sampling the state localShots
              // times.
              if (hasConditionals) {
                // Populate `counts` one shot at a time
                cudaq::sample_result counts;
                for (std::size_t shot = 0; shot < localShots; shot++) {
                  cudaq::ExecutionContext context("sample", 1);
                  context.hasConditionalsOnMeasureResults = true;
                  cudaq::getExecutionManager()->setExecutionContext(&context);
                  invokeJITKernel(localJIT[0], kernelName);
                  cudaq::getExecutionManager()->resetExecutionContext();
                  counts += context.result;
                }
                // Process `counts` and store into `results`
                for (auto &regName : counts.register_names()) {
                  results.emplace_back(counts.to_map(regName), regName);
                  results.back().sequentialData =
                      counts.sequential_data(regName);
                }
              }
            }

            for (std::size_t i = 0; i < codes.size(); i++) {
              cudaq::ExecutionContext context("sample", localShots);
              context.reorderIdx = reorderIdx;
              cudaq::getExecutionManager()->setExecutionContext(&context);
              invokeJITKernelAndRelease(localJIT[i], kernelName);
              cudaq::getExecutionManager()->resetExecutionContext();

              // If there are multiple codes, this is likely a spin_op.
              // If so, use the code name instead of the global register.
              if (codes.size() > 1) {
                results.emplace_back(context.result.to_map(), codes[i].name);
                results.back().sequentialData =
                    context.result.sequential_data();
              } else {
                // For each register, add the context results into result.
                for (auto &regName : context.result.register_names()) {
                  results.emplace_back(context.result.to_map(regName), regName);
                  results.back().sequentialData =
                      context.result.sequential_data(regName);
                }
              }
            }
            localJIT.clear();
            return cudaq::sample_result(results);
          }));

    } else {
      // Execute the codes produced in quake lowering
      // Allow developer to disable remote sending (useful for debugging IR)
      if (getEnvBool("DISABLE_REMOTE_SEND", false))
        return;
      else
        future = executor->execute(codes);
    }

    // Keep this asynchronous if requested
    if (executionContext->asyncExec) {
      executionContext->futureResult = future;
      return;
    }

    // Otherwise make this synchronous
    executionContext->result = future.get();
  }
};
} // namespace cudaq
