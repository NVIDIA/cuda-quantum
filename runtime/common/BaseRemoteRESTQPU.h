/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ArgumentConversion.h"
#include "common/CodeGenConfig.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/ExtraPayloadProvider.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/Resources.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "common/cudaq_fmt.h"
#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/operators.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "nvqpp_config.h"
#include "llvm/ADT/SmallSet.h"
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
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

namespace nvqir {
// QIR helper to retrieve the output log.
std::string_view getQirOutputLog();
cudaq::Resources *getResourceCounts();
bool isUsingResourceCounterSimulator();
} // namespace nvqir

namespace cudaq {

class BaseRemoteRESTQPU : public QPU {
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

  /// @brief Mapping of general key-values for backend configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Flag indicating whether we should emulate execution locally.
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

  std::tuple<mlir::ModuleOp, std::unique_ptr<mlir::MLIRContext>, void *>
  extractQuakeCodeAndContext(const std::string &kernelName, void *data) {
    auto context = cudaq::getOwningMLIRContext();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<mlir::ModuleOp>(quakeCode, context.get());
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    return std::make_tuple(m_module.release(), std::move(context), data);
  }

public:
  /// @brief The constructor
  BaseRemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    cudaq::initializeMLIR();
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

  /// @brief Return true if the current backend supports explicit measurements
  bool supportsExplicitMeasurements() override { return false; }

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

    // This check ensures that a kernel is not called whilst actively being
    // used for resource counting (implying that the kernel was somehow
    // invoked from inside the choice function). This check may want to
    // be expanded more broadly to ensure that the execution context is
    // always fully reset, implying the end of the invocation, being being
    // set again, signaling a new invocation.
    if (nvqir::isUsingResourceCounterSimulator() &&
        context->name != "resource-count")
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");

    CUDAQ_INFO("Remote Rest QPU setting execution context to {}",
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
    CUDAQ_INFO("Remote REST platform is targeting {}.", backend);

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
          CUDAQ_INFO("Decoded {} parameter from '{}' to '{}'", split[i],
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
    enablePassStatistics =
        getEnvBool("CUDAQ_MLIR_PASS_STATISTICS", enablePassStatistics);

    // If the very verbose enablePrintMLIREachPass flag is set, then
    // multi-threading must be disabled.
    if (enablePrintMLIREachPass) {
      disableMLIRthreading = true;
    }

    /// Once we know the backend, we should search for the configuration file
    /// from there we can get the URL/PORT and the required MLIR pass pipeline.
    std::string fileName = mutableBackend + std::string(".yml");
    auto configFilePath = platformPath / fileName;
    CUDAQ_INFO("Config file path = {}", configFilePath.string());
    std::ifstream configFile(configFilePath.string());
    std::string configYmlContents((std::istreambuf_iterator<char>(configFile)),
                                  std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configYmlContents.c_str());
    Input >> config;
    if (config.BackendConfig.has_value()) {
      const auto codeGenSpec = config.getCodeGenSpec(backendConfig);
      if (!codeGenSpec.empty()) {
        CUDAQ_INFO("Set codegen translation: {}", codeGenSpec);
        codegenTranslation = codeGenSpec;
        // Validate codegen configuration.
        parseCodeGenTranslation(codegenTranslation);
      }

      const std::string allowEarlyExitSetting =
          codegenTranslation.starts_with("qir-adaptive") ? "true" : "false";

      // 1. Apply all the target-agnostic high-level passes. If this is an
      // emulation and a noise model has been set, do not erase the noise
      // callbacks.
      if (emulate)
        passPipelineConfig += ",emul-jit-prep-pipeline{erase-noise=" +
                              std::string{noiseModel ? "false" : "true"} +
                              " allow-early-exit=" + allowEarlyExitSetting +
                              "}";
      else
        passPipelineConfig +=
            ",hw-jit-prep-pipeline{allow-early-exit=" + allowEarlyExitSetting +
            "}";

      // 2. Apply target-specific high-level passes from the .yml file, if any.
      if (!config.BackendConfig->JITHighLevelPipeline.empty()) {
        CUDAQ_INFO("Appending JIT high level pipeline: {}",
                   config.BackendConfig->JITHighLevelPipeline);
        passPipelineConfig += "," + config.BackendConfig->JITHighLevelPipeline;
      }

      // 3. Appply the target-agnostic deployment passes. Any additional
      // restructuring to get ready for decomposition.
      passPipelineConfig += ",jit-deploy-pipeline";

      // 4. Apply the target-specific mid-level passes. This decomposed quantum
      // gates for a specific target machine, etc.
      if (!config.BackendConfig->JITMidLevelPipeline.empty()) {
        CUDAQ_INFO("Appending JIT mid level pipeline: {}",
                   config.BackendConfig->JITMidLevelPipeline);
        passPipelineConfig += "," + config.BackendConfig->JITMidLevelPipeline;
      }

      // 5. Apply the target-agnostic finalization passes. This lowers the IR to
      // CFG form.
      passPipelineConfig += ",jit-finalize-pipeline";

      // 6. Apply the target-specific low-level passes.
      if (!config.BackendConfig->JITLowLevelPipeline.empty()) {
        CUDAQ_INFO("Appending JIT low level pipeline: {}",
                   config.BackendConfig->JITLowLevelPipeline);
        passPipelineConfig += "," + config.BackendConfig->JITLowLevelPipeline;
      }

      if (!config.BackendConfig->PostCodeGenPasses.empty()) {
        CUDAQ_INFO("Adding post-codegen lowering pipeline: {}",
                   config.BackendConfig->PostCodeGenPasses);
        postCodeGenPasses = config.BackendConfig->PostCodeGenPasses;
      }
    }

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
      CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
                 "pipeline to {}",
                 passPipelineConfig);
    }

    // Set the qpu name
    qpuName = mutableBackend;
    // Create the ServerHelper for this QPU and give it the backend config
    serverHelper = cudaq::registry::get<cudaq::ServerHelper>(qpuName);
    if (!serverHelper) {
      throw std::runtime_error("ServerHelper not found for target: " + qpuName);
    }

    serverHelper->initialize(backendConfig);
    serverHelper->updatePassPipeline(platformPath, passPipelineConfig);
    CUDAQ_INFO("Retrieving executor with name {}", qpuName);
    CUDAQ_INFO("Is this executor registered? {}",
               cudaq::registry::isRegistered<cudaq::Executor>(qpuName));
    executor = cudaq::registry::isRegistered<cudaq::Executor>(qpuName)
                   ? cudaq::registry::get<cudaq::Executor>(qpuName)
                   : std::make_unique<cudaq::Executor>();

    // Give the server helper to the executor
    executor->setServerHelper(serverHelper.get());

    // Construct the runtime target
    RuntimeTarget runtimeTarget;
    runtimeTarget.config = config;
    runtimeTarget.name = mutableBackend;
    runtimeTarget.description = config.Description;
    runtimeTarget.runtimeConfig = backendConfig;
    serverHelper->setRuntimeTarget(runtimeTarget);
  }

  /// Conditionally form an output_names JSON object if this was for QIR
  nlohmann::json formOutputNames(const std::string &codegenTranslation,
                                 mlir::ModuleOp moduleOp,
                                 const std::string &codeStr) {
    // Form an output_names mapping from codeStr
    nlohmann::json output_names;
    std::vector<char> bitcode;
    if (codegenTranslation.starts_with("qir")) {
      // decodeBase64 will throw a runtime exception if it fails
      if (llvm::decodeBase64(codeStr, bitcode)) {
        CUDAQ_INFO("Could not decode codeStr {}", codeStr);
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
    } else if (codegenTranslation.starts_with("qasm2")) {
      for (auto &op : moduleOp) {
        if (op.hasAttr(cudaq::entryPointAttrName) &&
            op.hasAttr("output_names")) {
          if (auto strAttr = op.getAttr(cudaq::opt::QIROutputNamesAttrName)
                                 .dyn_cast_or_null<mlir::StringAttr>()) {
            output_names = nlohmann::json::parse(strAttr.getValue());
            break;
          }
        }
      }
    }
    return output_names;
  }

  mlir::ModuleOp lowerQuakeCodeBuildModule(const std::string &,
                                           mlir::ModuleOp module,
                                           mlir::MLIRContext *,
                                           mlir::func::FuncOp);

  std::vector<cudaq::KernelExecution>
  lowerQuakeCodePart2(const std::string &kernelName, void *kernelArgs,
                      const std::vector<void *> &rawArgs,
                      mlir::ModuleOp m_module, mlir::MLIRContext *contextPtr,
                      void *updatedArgs) {
    // Extract the kernel name
    auto origFn = m_module.template lookupSymbol<mlir::func::FuncOp>(
        std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);

    auto moduleOp =
        lowerQuakeCodeBuildModule(kernelName, m_module, contextPtr, origFn);

    // Lambda to apply a specific pipeline to the given ModuleOp
    auto runPassPipeline = [&](const std::string &pipeline,
                               mlir::ModuleOp moduleOpIn) {
      mlir::PassManager pm(contextPtr);
      std::string errMsg;
      llvm::raw_string_ostream os(errMsg);
      CUDAQ_INFO("Pass pipeline for {} = {}", kernelName, pipeline);
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

    auto epFunc =
        moduleOp.template lookupSymbol<mlir::func::FuncOp>(origFn.getName());
    const bool isPython = moduleOp->hasAttr(runtime::pythonUniqueAttrName);
    if (!rawArgs.empty() || updatedArgs) {
      mlir::PassManager pm(contextPtr);
      if (isPython)
        detail::mergeAllCallableClosures(moduleOp, kernelName, rawArgs);

      // Mark all newly merged kernels private, and leave the entry point alone.
      for (auto &op : moduleOp)
        if (auto f = dyn_cast<mlir::func::FuncOp>(op))
          if (f != epFunc)
            f.setPrivate();

      if (!rawArgs.empty()) {
        CUDAQ_INFO("Run Argument Synth.\n");
        // For quantum devices, we generate a collection of `init` and
        // `num_qubits` functions and their substitutions created
        // from a kernel and arguments that generated a state argument.
        cudaq::opt::ArgumentConverter argCon(kernelName, moduleOp);
        argCon.gen(rawArgs);

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
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
        pm.addPass(
            cudaq::opt::createLambdaLifting({.constantPropagation = true}));
        // We must inline these lambda calls before apply specialization as it
        // does not perform control/adjoint specialization across function call
        // boundary.
        cudaq::opt::addAggressiveInlining(pm);
        pm.addPass(cudaq::opt::createApplySpecialization(
            {.constantPropagation = true}));
        cudaq::opt::addAggressiveInlining(pm);
        pm.addPass(mlir::createSymbolDCEPass());
      } else if (updatedArgs) {
        CUDAQ_INFO("Run Quake Synth.\n");
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

    if (emulate && executionContext && executionContext->name == "sample") {
      // Populate conditional measurement flag in the context.
      for (auto &artifact : moduleOp) {
        quake::detail::QuakeFunctionAnalysis analysis{&artifact};
        auto info = analysis.getAnalysisInfo();
        if (info.empty())
          continue;
        auto result = info[&artifact];
        if (result.hasConditionalsOnMeasure) {
          executionContext->hasConditionalsOnMeasureResults = true;
          break;
        }
      }
    }
    // Delay combining measurements for backends that cannot handle
    // subveqs and multiple measurements until we created the emulation code.
    auto combineMeasurements =
        passPipelineConfig.find("combine-measurements") != std::string::npos;
    if (emulate && combineMeasurements) {
      std::regex combine("(.*),([ ]*)combine-measurements(.*)");
      std::string replacement("$1$3");
      passPipelineConfig =
          std::regex_replace(passPipelineConfig, combine, replacement);
      CUDAQ_INFO("Delaying combine-measurements pass due to emulation. "
                 "Updating pipeline to {}",
                 passPipelineConfig);
    }

    runPassPipeline(passPipelineConfig, moduleOp);
    // We need to run resource counting preprocessing after the pass pipeline as
    // the pre-processing might change the IR structure (may interfere with
    // other passes).
    if (executionContext && executionContext->name == "resource-count") {
      // Each pass may run in a separate thread, so we have to make sure to
      // grab this reference in this thread
      auto resource_counts = nvqir::getResourceCounts();
      std::function<void(std::string, size_t, size_t)> f =
          [&](std::string gate, size_t nControls, size_t count) {
            CUDAQ_INFO("Appending: {}", gate);
            resource_counts->appendInstruction(gate, nControls, count);
          };
      cudaq::opt::ResourceCountPreprocessOptions opt{f};
      mlir::PassManager pm(moduleOp.getContext());
      pm.addNestedPass<mlir::func::FuncOp>(
          opt::createResourceCountPreprocess(opt));
      pm.addPass(mlir::createCanonicalizerPass());
      if (enablePrintMLIREachPass)
        pm.enableIRPrinting();
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error(
            "Could not successfully apply resource count preprocess.");
    }

    assert(
        moduleOp.template lookupSymbol<mlir::func::FuncOp>(epFunc.getName()) &&
        "Entry point function must survive the lowering pipeline.");
    std::vector<std::size_t> mapping_reorder_idx;
    if (auto mappingAttr = dyn_cast_if_present<mlir::ArrayAttr>(
            epFunc->getAttr("mapping_reorder_idx"))) {
      mapping_reorder_idx.resize(mappingAttr.size());
      std::transform(mappingAttr.begin(), mappingAttr.end(),
                     mapping_reorder_idx.begin(), [](mlir::Attribute attr) {
                       return mlir::cast<mlir::IntegerAttr>(attr).getInt();
                     });
    }

    if (executionContext) {
      if (executionContext->name == "sample") {
        executionContext->reorderIdx = mapping_reorder_idx;
        // No need to add measurements only to remove them eventually
        if (postCodeGenPasses.find("remove-measurements") == std::string::npos)
          runPassPipeline("func.func(add-measurements)", moduleOp);
      } else {
        executionContext->reorderIdx.clear();
      }
    }

    std::vector<std::pair<std::string, mlir::ModuleOp>> modules;
    // Apply observations if necessary
    if (executionContext && executionContext->name == "observe") {
      mapping_reorder_idx.clear();
      runPassPipeline("canonicalize,cse", moduleOp);
      cudaq::spin_op &spin = executionContext->spin.value();
      for (const auto &term : spin) {
        if (term.is_identity())
          continue;

        // Get the ansatz
        [[maybe_unused]] auto ansatz =
            moduleOp.template lookupSymbol<mlir::func::FuncOp>(
                cudaq::runtime::cudaqGenPrefixName + kernelName);
        assert(ansatz && "could not find the ansatz kernel");

        // Create a new Module to clone the ansatz into it
        auto tmpModuleOp = moduleOp.clone();

        // Create the pass manager, add the quake observe ansatz pass and run it
        // followed by the canonicalizer
        mlir::PassManager pm(contextPtr);
        pm.addNestedPass<mlir::func::FuncOp>(
            cudaq::opt::createObserveAnsatzPass(
                term.get_binary_symplectic_form()));
        if (disableMLIRthreading || enablePrintMLIREachPass)
          tmpModuleOp.getContext()->disableMultithreading();
        if (enablePrintMLIREachPass)
          pm.enableIRPrinting();
        if (failed(pm.run(tmpModuleOp)))
          throw std::runtime_error("Could not apply measurements to ansatz.");
        // The full pass pipeline was run above, but the ansatz pass can
        // introduce gates that aren't supported by the backend, so we need to
        // re-run the gate set mapping if that existed in the original pass
        // pipeline.
        auto csvSplit = cudaq::split(passPipelineConfig, ',');
        for (auto &pass : csvSplit)
          if (pass.ends_with("-gate-set-mapping"))
            runPassPipeline(pass, tmpModuleOp);
        if (!emulate && combineMeasurements)
          runPassPipeline("func.func(combine-measurements)", tmpModuleOp);
        modules.emplace_back(term.get_term_id(), tmpModuleOp);
      }
    } else {
      modules.emplace_back(kernelName, moduleOp);
    }

    if (emulate ||
        (executionContext && executionContext->name == "resource-count")) {
      // If we are in emulation mode, we need to first get a full QIR
      // representation of the code. Then we'll map to an LLVM Module, create a
      // JIT ExecutionEngine pointer and use that for execution
      for (auto &[name, module] : modules) {
        auto clonedModule = module.clone();
        jitEngines.emplace_back(
            cudaq::createQIRJITEngine(clonedModule, codegenTranslation));
      }
    }

    if (emulate && combineMeasurements)
      for (auto &[name, module] : modules)
        runPassPipeline("func.func(combine-measurements)", module);

    // Get the code gen translation
    auto translation = cudaq::getTranslation(codegenTranslation);

    // Apply user-specified codegen
    std::vector<cudaq::KernelExecution> codes;
    for (auto &[name, moduleOpI] : modules) {
      std::string codeStr;
      llvm::raw_string_ostream outStr(codeStr);
      if (disableMLIRthreading)
        moduleOpI.getContext()->disableMultithreading();
      if (codegenTranslation.starts_with("qir")) {
        if (failed(translation(moduleOpI, codegenTranslation, outStr,
                               postCodeGenPasses, printIR,
                               enablePrintMLIREachPass, enablePassStatistics)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      } else {
        if (failed(translation(moduleOpI, outStr, postCodeGenPasses, printIR,
                               enablePrintMLIREachPass, enablePassStatistics)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      }

      // Form an output_names mapping from codeStr
      nlohmann::json j =
          formOutputNames(codegenTranslation, moduleOpI, codeStr);

      codes.emplace_back(name, codeStr, j, mapping_reorder_idx);
    }

    return codes;
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
    return lowerQuakeCodePart2(kernelName, kernelArgs, rawArgs, m_module,
                               contextPtr.get(), updatedArgs);
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

  // Here the quake code is passed to us (via a ModuleOp), so unlike the other
  // lowerQuakeCode() member functions there is no need to surf dictionaries for
  // strings of code to assemble. We have to make sure that this MLIRContext is
  // not destroyed however, since it may hold an unknown number of other
  // ModuleOps.
  // Unchecked assumption: \p module is referentially unique (within the scope
  // of this launch instance) and disposable. It can be modified by this call in
  // any way necessary without breaking some other kernel launch.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, mlir::ModuleOp module,
                 const std::vector<void *> &rawArgs) {
    return lowerQuakeCodePart2(kernelName, nullptr, rawArgs, module,
                               module.getContext(), nullptr);
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    // Get the Quake code, lowered according to config file.
    auto codes = lowerQuakeCode(kernelName, rawArgs);
    completeLaunchKernel(kernelName, std::move(codes));
  }

  /// @brief Launch the kernel. Extract the Quake code and lower to the
  /// representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as asynchronous or
  /// synchronous invocation.
  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    // Get the Quake code, lowered according to config file.
    // FIXME: For python, we reach here with rawArgs being empty and args having
    // the arguments. Python should be using the streamlined argument synthesis,
    // but apparently it isn't. This works around that bug.
    auto codes = rawArgs.empty() ? lowerQuakeCode(kernelName, args)
                                 : lowerQuakeCode(kernelName, rawArgs);
    completeLaunchKernel(kernelName, std::move(codes));

    // NB: Kernel should/will never return dynamic results.
    return {};
  }

  KernelThunkResultType launchModule(const std::string &kernelName,
                                     mlir::ModuleOp module,
                                     const std::vector<void *> &rawArgs,
                                     mlir::Type resTy) override {
    CUDAQ_INFO("launching remote rest kernel via module ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    completeLaunchKernel(kernelName,
                         lowerQuakeCode(kernelName, module, rawArgs));
    return {};
  }

  void *specializeModule(const std::string &kernelName, mlir::ModuleOp module,
                         const std::vector<void *> &rawArgs, mlir::Type resTy,
                         void *cachedEngine) override {
    CUDAQ_INFO("specializing remote rest kernel via module ({})", kernelName);
    throw std::runtime_error(
        "NYI: Remote rest execution via Python/C++ interop.");
    return nullptr;
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

    if (executionContext->name == "resource-count") {
      assert(jitEngines.size() == 1);
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
    const bool isObserve =
        executionContext && executionContext->name == "observe";
    const bool isRun = executionContext && executionContext->name == "run";

    // If emulation requested, then just grab the function and invoke it with
    // the simulator
    cudaq::details::future future;
    if (emulate) {

      // Fetch the thread-specific seed outside and then pass it inside.
      std::size_t seed = cudaq::get_random_seed();

      // Launch the execution of the simulated jobs asynchronously
      future = cudaq::details::future(std::async(
          std::launch::async,
          [&, codes, localShots, kernelName, seed, isObserve, isRun,
           reorderIdx = executionContext->reorderIdx,
           localJIT = std::move(jitEngines)]() mutable -> cudaq::sample_result {
            std::vector<cudaq::ExecutionResult> results;

            // If seed is 0, then it has not been set.
            if (seed > 0)
              cudaq::set_random_seed(seed);

            const bool hasConditionals =
                executionContext
                    ? executionContext->hasConditionalsOnMeasureResults
                    : false;

            if (hasConditionals && isObserve)
              throw std::runtime_error("error: spin_ops not yet supported with "
                                       "kernels containing conditionals");
            if (isRun || hasConditionals) {
              // Validate the execution logic: cudaq::run and cudaq::sample on
              // conditional kernels should only generate one JIT'ed kernel.
              assert(localJIT.size() == 1);
              executor->setShots(1); // run one shot at a time

              // If this is adaptive profile and the kernel has conditionals or
              // executed via cudaq::run, then you have to run the code
              // localShots times instead of running the kernel once and
              // sampling the state localShots times.

              // If not executed via cudaq::run, we populate `counts` one shot
              // at a time.
              cudaq::sample_result counts;
              for (std::size_t shot = 0; shot < localShots; shot++) {
                cudaq::ExecutionContext context("sample", 1);
                context.hasConditionalsOnMeasureResults = true;
                if (!isRun)
                  cudaq::getExecutionManager()->setExecutionContext(&context);

                invokeJITKernel(localJIT[0], kernelName);
                if (!isRun) {
                  cudaq::getExecutionManager()->resetExecutionContext();
                  counts += context.result;
                }
              }
              if (!isRun) {
                // Process `counts` and store into `results`
                for (auto &regName : counts.register_names()) {
                  results.emplace_back(counts.to_map(regName), regName);
                  results.back().sequentialData =
                      counts.sequential_data(regName);
                }
              } else {
                // Get QIR output log
                const auto qirOutputLog = nvqir::getQirOutputLog();
                executionContext->invocationResultBuffer.assign(
                    qirOutputLog.begin(), qirOutputLog.end());
              }
            } else {
              // Otherwise, this is a non-adaptive sampling or observe.
              // We run the kernel(s) (multiple kernels if this is a multi-term
              // observe) one time each.
              for (std::size_t i = 0; i < codes.size(); i++) {
                cudaq::ExecutionContext context("sample", localShots);
                context.reorderIdx = reorderIdx;
                cudaq::getExecutionManager()->setExecutionContext(&context);
                invokeJITKernel(localJIT[i], kernelName);
                cudaq::getExecutionManager()->resetExecutionContext();

                if (isObserve) {
                  // Use the code name instead of the global register.
                  results.emplace_back(context.result.to_map(), codes[i].name);
                  results.back().sequentialData =
                      context.result.sequential_data();
                } else {
                  // For each register, add the context results into result.
                  for (auto &regName : context.result.register_names()) {
                    results.emplace_back(context.result.to_map(regName),
                                         regName);
                    results.back().sequentialData =
                        context.result.sequential_data(regName);
                  }
                }
              }
            }

            // Clean up the JIT engines. This functor owns these engine
            // instances.
            for (auto *jitEngine : localJIT)
              delete jitEngine;
            localJIT.clear();
            return cudaq::sample_result(results);
          }));

    } else {
      // Execute the codes produced in quake lowering
      // Allow developer to disable remote sending (useful for debugging IR)
      if (getEnvBool("DISABLE_REMOTE_SEND", false))
        return;
      // Cannot be observe and run at the same time
      assert(!isObserve || !isRun);
      const cudaq::details::ExecutionContextType execType =
          isRun       ? cudaq::details::ExecutionContextType::run
          : isObserve ? cudaq::details::ExecutionContextType::observe
                      : cudaq::details::ExecutionContextType::sample;

      future = executor->execute(codes, execType,
                                 &executionContext->invocationResultBuffer);
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

inline mlir::ModuleOp BaseRemoteRESTQPU::lowerQuakeCodeBuildModule(
    const std::string &kernelName, mlir::ModuleOp m_module,
    mlir::MLIRContext *contextPtr, mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::func::FuncOp> newFuncOpsWithDefinitions;
  llvm::SmallSet<std::string, 4> deviceCallCallees;
  // For every declaration without a definition, we need to try to find the
  // function in the Quake registry and copy the functions into this module.
  m_module.walk([&](mlir::func::FuncOp funcOp) {
    if (!funcOp.isDeclaration()) {
      // Skipping function because it already has a definition.
      return mlir::WalkResult::advance();
    }
    // Definition doesn't exist, so we need to find it in the Quake
    // registry.
    mlir::StringRef fullFuncName = funcOp.getName();
    mlir::StringRef kernelName = [fullFuncName]() {
      mlir::StringRef retVal = fullFuncName;
      // TODO - clean this up to not have to do this. Considering the
      // module's map, or cudaq::details::getKernelName(). But make sure it
      // works for standard C++ functions.

      // Only get the portion before the first ".".
      if (auto ix = fullFuncName.find("."); ix != mlir::StringRef::npos)
        retVal = fullFuncName.substr(0, ix);
      // Also strip out __nvqpp_mlirgen__function_ from the beginning of the
      // function name.
      if (retVal.starts_with(cudaq::runtime::cudaqGenPrefixName)) {
        retVal = retVal.substr(cudaq::runtime::cudaqGenPrefixLength);
      }
      return retVal;
    }();
    std::string quakeCode =
        kernelName.empty() ? ""
                           : cudaq::get_quake_by_name(kernelName.str(),
                                                      /*throwException=*/false);
    if (quakeCode.empty()) {
      // Skipping function because it does not have a quake code.
      return mlir::WalkResult::advance();
    }
    auto tmp_module = parseSourceString<mlir::ModuleOp>(quakeCode, contextPtr);
    auto tmpFuncOpWithDefinition =
        tmp_module->lookupSymbol<mlir::func::FuncOp>(fullFuncName);
    auto newNameAttr = mlir::StringAttr::get(m_module.getContext(),
                                             fullFuncName.str() + ".stitch");
    auto clonedFunc = tmpFuncOpWithDefinition.clone();
    clonedFunc.setName(newNameAttr);
    mlir::SymbolTable symTable(m_module);
    symTable.insert(clonedFunc);
    newFuncOpsWithDefinitions.push_back(clonedFunc);

    if (failed(mlir::SymbolTable::replaceAllSymbolUses(
            funcOp.getOperation(), newNameAttr, m_module.getOperation()))) {
      throw std::runtime_error(fmt::format(
          "Failed to replace symbol uses for function {}", fullFuncName.str()));
    }
    return mlir::WalkResult::advance();
  });

  // For each one of the added functions, we need to traverse them to find
  // device calls (in order to create declarations for them)
  for (auto &funcOp : newFuncOpsWithDefinitions) {
    mlir::OpBuilder builder(m_module);
    builder.setInsertionPointToStart(m_module.getBody());
    funcOp.walk([&](cudaq::cc::DeviceCallOp deviceCall) {
      auto calleeName = deviceCall.getCallee();
      // If the callee is already in the symbol table, nothing to do.
      if (m_module.lookupSymbol<mlir::func::FuncOp>(calleeName))
        return;

      // Otherwise, we need to create a declaration for the callback
      // function.
      auto argTypes = deviceCall.getArgs().getTypes();
      auto resTypes = deviceCall.getResultTypes();
      auto funcType = builder.getFunctionType(argTypes, resTypes);

      // Create a *declaration* (no body) for the callback function.
      [[maybe_unused]] auto decl = builder.create<mlir::func::FuncOp>(
          deviceCall.getLoc(), calleeName, funcType);
      decl.setPrivate();
      deviceCallCallees.insert(calleeName.str());
    });
  }

  // Create a new Module to clone the function into
  auto location = mlir::FileLineColLoc::get(contextPtr, "<builder>", 1, 1);
  mlir::ImplicitLocOpBuilder builder(location, contextPtr);

  // FIXME this should be added to the builder.
  if (!func->hasAttr(cudaq::entryPointAttrName))
    func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
  auto moduleOp = builder.create<mlir::ModuleOp>();
  moduleOp->setAttrs(m_module->getAttrDictionary());
  auto mangledNameMap = m_module->getAttrOfType<mlir::DictionaryAttr>(
      cudaq::runtime::mangledNameMap);

  for (auto &op : m_module.getOps()) {
    // Add any global symbols, including global constant arrays. Global
    // constant arrays can be created during compilation,
    // `lift-array-alloc`, `argument-synthesis`, `quake-synthesizer`, and
    // `get-concrete-matrix` passes.
    if (auto lfunc = dyn_cast<mlir::func::FuncOp>(op)) {
      bool skip = lfunc.getName().ends_with(".thunk");
      if (!skip && !deviceCallCallees.contains(lfunc.getName().str()))
        for (auto &entry : mangledNameMap)
          if (lfunc.getName() ==
              cast<mlir::StringAttr>(entry.getValue()).getValue()) {
            skip = true;
            break;
          }
      if (!skip) {
        auto clonedFunc = lfunc.clone();
        if (clonedFunc.getName() != func.getName())
          clonedFunc.setPrivate();
        moduleOp.push_back(std::move(clonedFunc));
      }
    } else {
      moduleOp.push_back(op.clone());
    }
  }
  return moduleOp;
}
} // namespace cudaq
