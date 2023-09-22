/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Executor.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Plugin.h"
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
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

using namespace mlir;

namespace cudaq {
std::string get_quake_by_name(const std::string &);
} // namespace cudaq

namespace {

/// @brief The RemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA Quantum kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service. Specific
/// details about JSON payloads are abstracted via an abstract type called
/// ServerHelper, which is meant to be subtyped by each provided remote QPU
/// service. Moreover, this QPU handles launching kernels under a number of
/// Execution Contexts, including sampling and observation via synchronous or
/// asynchronous client invocations. This type should enable both QIR-based
/// backends as well as those that take OpenQASM2 as input.
class RemoteRESTQPU : public cudaq::QPU {
protected:
  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The Pass pipeline string, configured by the
  /// QPU config file in the platform path.
  std::string passPipelineConfig = "canonicalize";

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Name of codegen translation (e.g. "qir", "qasm2", "iqm")
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
  /// -mlir-disable-threading for cudaq-opt.
  bool disableMLIRthreading = false;

  /// @brief Flag indicating whether we should enable MLIR printing before and
  /// after each pass. This is similar to (-mlir-print-ir-before-all and
  /// -mlir-print-ir-after-all) in cudaq-opt.
  bool enablePrintMLIREachPass = false;

  /// @brief If we are emulating locally, keep track
  /// of JIT engines for invoking the kernels.
  std::vector<ExecutionEngine *> jitEngines;

  /// @brief Invoke the kernel in the JIT engine and then delete the JIT engine.
  void invokeJITKernelAndRelease(ExecutionEngine *jit,
                                 const std::string &kernelName) {
    auto funcPtr = jit->lookup(std::string("__nvqpp__mlirgen__") + kernelName);
    if (!funcPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get kernelReg function.");
    }
    reinterpret_cast<void (*)()>(*funcPtr)();
    // We're done, delete the pointer.
    delete jit;
  }

  /// @brief Helper function to get boolean environment variable
  bool getEnvBool(const char *envName, bool defaultVal = false) {
    if (auto envVal = std::getenv(envName)) {
      std::string tmp(envVal);
      std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (tmp == "1" || tmp == "on" || tmp == "true" || tmp == "yes")
        return true;
    }
    return defaultVal;
  }

public:
  /// @brief The constructor
  RemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<cudaq::Executor>();
  }

  RemoteRESTQPU(RemoteRESTQPU &&) = delete;
  virtual ~RemoteRESTQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  /// @return
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supportsConditionalFeedback() override { return false; }

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
  /// CUDA Quantum installation) and extract MLIR lowering pipelines and
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
      for (std::size_t i = 1; i < split.size(); i += 2)
        backendConfig.insert({split[i], split[i + 1]});
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

    /// Once we know the backend, we should search for the config file
    /// from there we can get the URL/PORT and the required MLIR pass
    /// pipeline.
    std::string fileName = mutableBackend + std::string(".config");
    auto configFilePath = platformPath / fileName;
    cudaq::info("Config file path = {}", configFilePath.string());
    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());

    // Loop through the file, extract the pass pipeline and CODEGEN Type
    auto lines = cudaq::split(configContents, '\n');
    std::regex pipeline("^PLATFORM_LOWERING_CONFIG\\s*=\\s*\"(\\S+)\"");
    std::regex emissionType("^CODEGEN_EMISSION\\s*=\\s*(\\S+)");
    std::regex postCodeGen("^POST_CODEGEN_PASSES\\s*=\\s*\"(\\S+)\"");
    std::smatch match;
    for (const std::string &line : lines) {
      if (std::regex_search(line, match, pipeline)) {
        cudaq::info("Appending lowering pipeline: {}", match[1].str());
        passPipelineConfig += "," + match[1].str();
      } else if (std::regex_search(line, match, emissionType)) {
        codegenTranslation = match[1].str();
      } else if (std::regex_search(line, match, postCodeGen)) {
        cudaq::info("Adding post-codegen lowering pipeline: {}",
                    match[1].str());
        postCodeGenPasses = match[1].str();
      }
    }

    // Set the qpu name
    qpuName = mutableBackend;

    // Create the ServerHelper for this QPU and give it the backend config
    serverHelper = cudaq::registry::get<cudaq::ServerHelper>(qpuName);
    serverHelper->initialize(backendConfig);

    // Give the server helper to the executor
    executor->setServerHelper(serverHelper.get());
  }

  /// @brief Conditionally form an output_names JSON object if this was for QIR
  nlohmann::json formOutputNames(const std::string &codegenTranslation,
                                 const std::string &codeStr) {
    // Form an output_names mapping from codeStr
    nlohmann::json output_names;
    std::vector<char> bitcode;
    if (codegenTranslation == "qir") {
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
  /// lowering process is controllable via the platforms/BACKEND.config file for
  /// this targeted backend.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs) {

    auto contextPtr = cudaq::initializeMLIR();
    MLIRContext &context = *contextPtr.get();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, &context);
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    // Extract the kernel name
    auto func = m_module->lookupSymbol<mlir::func::FuncOp>(
        std::string("__nvqpp__mlirgen__") + kernelName);

    // Create a new Module to clone the function into
    auto location = FileLineColLoc::get(&context, "<builder>", 1, 1);
    ImplicitLocOpBuilder builder(location, &context);

    // FIXME this should be added to the builder.
    if (!func->hasAttr(cudaq::entryPointAttrName))
      func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
    auto moduleOp = builder.create<ModuleOp>();
    moduleOp.push_back(func.clone());

    // Lambda to apply a specific pipeline to the given ModuleOp
    auto runPassPipeline = [&](const std::string &pipeline,
                               ModuleOp moduleOpIn) {
      PassManager pm(&context);
      std::string errMsg;
      llvm::raw_string_ostream os(errMsg);
      cudaq::info("Pass pipeline for {} = {}", kernelName, pipeline);
      if (failed(parsePassPipeline(pipeline, pm, os)))
        throw std::runtime_error(
            "Remote rest platform failed to add passes to pipeline (" + errMsg +
            ").");
      if (failed(pm.run(moduleOpIn)))
        throw std::runtime_error("Remote rest platform Quake lowering failed.");
    };

    if (kernelArgs) {
      cudaq::info("Run Quake Synth.\n");
      PassManager pm(&context);
      pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, kernelArgs));
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error("Could not successfully apply quake-synth.");
    }

    // Run the config-specified pass pipeline
    runPassPipeline(passPipelineConfig, moduleOp);

    std::vector<std::pair<std::string, ModuleOp>> modules;
    // Apply observations if necessary
    if (executionContext && executionContext->name == "observe") {

      cudaq::spin_op &spin = *executionContext->spin.value();
      for (const auto &term : spin) {
        if (term.is_identity())
          continue;

        // Get the ansatz
        auto ansatz = moduleOp.lookupSymbol<func::FuncOp>(
            std::string("__nvqpp__mlirgen__") + kernelName);

        // Create a new Module to clone the ansatz into it
        auto tmpModuleOp = builder.create<ModuleOp>();
        tmpModuleOp.push_back(ansatz.clone());

        // Extract the binary symplectic encoding
        auto [binarySymplecticForm, coeffs] = term.get_raw_data();

        // Create the pass manager, add the quake observe ansatz pass
        // and run it followed by the canonicalizer
        PassManager pm(&context);
        OpPassManager &optPM = pm.nest<func::FuncOp>();
        optPM.addPass(
            cudaq::opt::createQuakeObserveAnsatzPass(binarySymplecticForm[0]));
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
        jitEngines.emplace_back(cudaq::createQIRJITEngine(clonedModule));
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

      codes.emplace_back(name, codeStr, j);
    }
    return codes;
  }

  /// @brief Launch the kernel. Extract the Quake code and lower to
  /// the representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as async or sync
  /// invocation.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error("Remote rest execution can only be performed "
                               "via cudaq::sample() or cudaq::observe().");

    // Get the Quake code, lowered according to config file.
    auto codes = lowerQuakeCode(kernelName, args);
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
           localJIT = std::move(jitEngines)]() mutable -> cudaq::sample_result {
            std::vector<cudaq::ExecutionResult> results;

            // If seed is 0, then it has not been set.
            if (seed > 0)
              cudaq::set_random_seed(seed);

            for (std::size_t i = 0; i < codes.size(); i++) {
              cudaq::ExecutionContext context("sample", localShots);
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
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteRESTQPU, remote_rest)
