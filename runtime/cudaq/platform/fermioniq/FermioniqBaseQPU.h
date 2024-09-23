/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ArgumentConversion.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
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

class FermioniqBaseQPU : public cudaq::QPU {
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

  /// @brief Flag indicating whether we should enable MLIR pass statistics
  /// to be printed. This is similar to `-mlir-pass-statistics` in `cudaq-opt`
  bool enablePassStatistics = false;

  /// @brief If we are emulating locally, keep track
  /// of JIT engines for invoking the kernels.
  std::vector<mlir::ExecutionEngine *> jitEngines;

  /// @brief Invoke the kernel in the JIT engine
  void invokeJITKernel(mlir::ExecutionEngine *jit,
                       const std::string &kernelName) {
    auto funcPtr = jit->lookup(std::string(cudaq::runtime::cudaqGenPrefixName) +
                               kernelName);
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
  FermioniqBaseQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<cudaq::Executor>();
  }

  FermioniqBaseQPU(FermioniqBaseQPU &&) = delete;
  virtual ~FermioniqBaseQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  /// @return
  bool isSimulator() override { return false; }

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
  virtual bool isRemote() override { return true; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return false; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  void setNoiseModel(const cudaq::noise_model *model) override {
    throw std::runtime_error(
          "Noise modeling is not allowed on this backend");
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

    cudaq::debug("reset execution context");

    // set the pre-computed expectation value.
    if (executionContext->name == "observe") {
      auto expectation =
          executionContext->result.expectation(GlobalRegisterName);
      cudaq::debug("got expectation: {}", expectation);
      executionContext->expectationValue =
          executionContext->result.expectation(GlobalRegisterName);
    }
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

    // Print the IR if requested
    printIR = getEnvBool("CUDAQ_DUMP_JIT_IR", printIR);

    // Get additional debug values
    disableMLIRthreading =
        getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", disableMLIRthreading);
    enablePrintMLIREachPass =
        getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", enablePrintMLIREachPass);
    enablePassStatistics =
        getEnvBool("CUDAQ_MLIR_PASS_STATISTICS", enablePassStatistics);

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

  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs) {
    return lowerQuakeCode(kernelName, kernelArgs, {});
  }

  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName,
                 const std::vector<void *> &rawArgs) {
    return lowerQuakeCode(kernelName, nullptr, rawArgs);
  }

  /// @brief Extract the Quake representation for the given kernel name and
  /// lower it to the code format required for the specific backend. The
  /// lowering process is controllable via the configuration file in the
  /// platform directory for the targeted backend.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs,
                 const std::vector<void *> &rawArgs) {

    auto [m_module, contextPtr, updatedArgs] =
        extractQuakeCodeAndContext(kernelName, kernelArgs);

    mlir::MLIRContext &context = *contextPtr;

    // Extract the kernel name
    auto func = m_module.lookupSymbol<mlir::func::FuncOp>(
        std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);

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

    if (!rawArgs.empty() || updatedArgs) {
      mlir::PassManager pm(&context);
      if (!rawArgs.empty()) {
        cudaq::info("Run Argument Synth.\n");
        opt::ArgumentConverter argCon(kernelName, moduleOp, false);
        argCon.gen(rawArgs);
        std::string kernName = cudaq::runtime::cudaqGenPrefixName + kernelName;
        mlir::SmallVector<mlir::StringRef> kernels = {kernName};
        std::string substBuff;
        llvm::raw_string_ostream ss(substBuff);
        ss << argCon.getSubstitutionModule();
        mlir::SmallVector<mlir::StringRef> substs = {substBuff};
        pm.addNestedPass<mlir::func::FuncOp>(
            opt::createArgumentSynthesisPass(kernels, substs));
      } else if (updatedArgs) {
        cudaq::info("Run Quake Synth.\n");
        pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, updatedArgs));
      }
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
        std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);
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
    modules.emplace_back(kernelName, moduleOp);

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
                               enablePrintMLIREachPass, enablePassStatistics)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      }

      nlohmann::json j;
      // Form an output_names mapping from codeStr
      if (executionContext->name == "observe") {
        j = "[[[0,[0, \"expectation%0\"]]]]"_json;
      } else {
        j = formOutputNames(codegenTranslation, codeStr);
      }

      cudaq::debug("Output names: {}", j.dump());

      if (executionContext->name == "observe") {

        auto spin = executionContext->spin.value();

        auto user_data = nlohmann::json::object();

        auto obs = nlohmann::json::array();

        spin->for_each_term([&](spin_op &term) {
          auto spin_op = nlohmann::json::object();

          auto terms = nlohmann::json::array();

          auto termStr = term.to_string(false);

          terms.push_back(termStr);

          auto coeff = term.get_coefficient();
          auto coeff_str = fmt::format("{}{}{}j", coeff.real(),
                                       coeff.imag() < 0.0 ? "-" : "+",
                                       std::fabs(coeff.imag()));

          terms.push_back(coeff_str);

          obs.push_back(terms);
        });

        user_data["observable"] = obs;

        codes.emplace_back(name, codeStr, j, mapping_reorder_idx, user_data);
      } else {
        codes.emplace_back(name, codeStr, j, mapping_reorder_idx);
      }
    }

    cleanupContext(contextPtr);
    return codes;
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::draw().");

    // Get the Quake code, lowered according to config file.
    auto codes = lowerQuakeCode(kernelName, rawArgs);
    completeLaunchKernel(kernelName, std::move(codes));
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
    completeLaunchKernel(kernelName, std::move(codes));
  }

  void completeLaunchKernel(const std::string &kernelName,
                            std::vector<cudaq::KernelExecution> &&codes) {

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
    
    // Execute the codes produced in quake lowering
    // Allow developer to disable remote sending (useful for debugging IR)
    if (getEnvBool("DISABLE_REMOTE_SEND", false))
      return;
    else
      future = executor->execute(codes);
  

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
