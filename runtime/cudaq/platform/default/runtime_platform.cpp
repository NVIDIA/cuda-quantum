/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/RuntimeMLIR.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
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
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qudit.h"
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
#include <regex>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>

namespace cudaq {

/// Holds all the values a runtime may want to know when executing QIR.
class runtime_config {
public:
  runtime_config() = default;
  runtime_config(std::size_t shots) : shots(shots) {}

  std::optional<std::size_t> shots;
};

class supported_llvm_instructions {
public:
  supported_llvm_instructions() = default;

  bool ret = false;
  bool br = false;
  bool switch_ = false;
  bool indirectbr = false;
  bool invoke = false;
  bool callbr = false;
  bool resume = false;
  bool catchswitch = false;
  bool catchret = false;
  bool cleanupret = false;
  bool unreachable = false;

  bool fneg = false;

  bool add = false;
  bool sub = false;
  bool mul = false;
  bool udiv = false;
  bool sdiv = false;
  bool fadd = false;
  bool fsub = false;
  bool fmul = false;
  bool fdiv = false;
  bool urem = false;
  bool srem = false;
  bool frem = false;

  bool shl = false;
  bool lshr = false;
  bool ashr = false;
  bool and_ = false;
  bool or_ = false;
  bool xor_ = false;

  bool extractelement = false;
  bool insertelement = false;
  bool shufflevector = false;

  bool extractvalue = false;
  bool insertvalue = false;

  bool alloc = false;  
  bool load = false;
  bool store = false;
  bool fence = false;
  bool cmpxchg = false;
  bool atomicrmw = false;
  bool getelementptr = false;
  
  bool trunc = false;
  bool zext = false;
  bool sext = false;
  bool fptrunc = false;
  bool fpext = false;
  bool fptoui = false;
  bool fptosi = false;
  bool uitofp = false;
  bool sitofp = false;
  bool ptrtoint = false;
  bool inttoptr = false;
  bool bitcase = false;
  bool addrspacecast = false;

  bool icmp = false;
  bool fcmp = false;
  bool phi = false;
  bool select = false;
  bool freeze = false;
  bool call = false;
  bool va_arg = false;
  bool landingpad = false;
  bool catchpad = false;
  bool cleanuppad = false;

  bool supports_floats() {
    return fadd && fsub && fmul && fdiv;
  }

  bool supports_integers() {
    return add && sub && mul && udiv && sdiv;
  }

  bool supports_pointers() {
    return getelementptr && alloc;
  }

  bool supports_classic_memory() {
    return store && load;
  }

  // ... and more composites.
};

class supported_qir_instructions {
public:
  supported_qir_instructions() = default;

  bool qis_r_body = false;
  bool qis_r_ctl = false;
  bool qis_r_adj = false;
  bool qis_r_ctladj = false;

  bool qis_rx_body = false;
  bool qis_rx_ctl = false;
  bool qis_rx_adj = false;
  bool qis_rx_ctladj = false;

  bool qis_x_body = false;
  bool qis_x_ctl = false;
  bool qis_x_adj = false;
  bool qis_x_ctladj = false;

  bool qis_ry_body = false;
  bool qis_ry_ctl = false;
  bool qis_ry_adj = false;
  bool qis_ry_ctladj = false;
  
  bool qis_y_body = false;
  bool qis_y_ctl = false;
  bool qis_y_adj = false;
  bool qis_y_ctladj = false;

  bool qis_rz_body = false;
  bool qis_rz_ctl = false;
  bool qis_rz_adj = false;
  bool qis_rz_ctladj = false;

  bool qis_z_body = false;
  bool qis_z_ctl = false;
  bool qis_z_adj = false;
  bool qis_z_ctladj = false;

  bool qis_s_body = false;
  bool qis_s_ctl = false;
  bool qis_s_adj = false;
  bool qis_s_ctladj = false;
  
  bool qis_t_body = false;
  bool qis_t_ctl = false;
  bool qis_t_adj = false;
  bool qis_t_ctladj = false;

  bool qis_ccx_body = false;
  bool qis_cx_body = false;
  bool qis_h_body = false;
  bool qis_h_ctl = false;
  bool qis_mz_body = false;
  bool qis_m_body = false;
  bool qis_reset_body = false;

  bool rt_initialize = false;
  bool rt_tuple_record_output = false;
  bool rt_array_record_output = false;
  bool rt_result_record_output = false;
  
  bool rt_result_get_zero = false;
  bool rt_result_get_one = false;
  bool rt_result_equal = false;

  bool rt_string_create = false;
  bool rt_string_get_data = false;
  bool rt_string_get_length = false;
  bool rt_string_concatenate = false;
  bool rt_string_equal = false;

  bool rt_int_to_string = false;
  bool rt_double_to_string = false;
  bool rt_bool_to_string = false;
  bool rt_result_to_string = false;
  bool rt_pauli_to_string = false;
  bool rt_qubit_to_string = false;
  bool rt_range_to_string = false;

  bool rt_tuple_create = false;
  bool rt_tuple_copy = false;
 
  bool rt_array_copy = false;
  bool rt_array_concatenate = false;
  bool rt_array_get_dim = false;
  bool rt_array_project = false;
  bool rt_array_create = false;   
  bool rt_array_create_1d = false;
  bool rt_array_slice = false;
  bool rt_array_slice_1d = false;
  bool rt_array_get_size = false;
  bool rt_array_get_size_1d = false;
  bool rt_array_get_element_ptr = false;
  bool rt_array_get_element_ptr_1d = false;

  bool rt_callable_create = false;
  bool rt_callable_copy = false;
  bool rt_callable_invoke = false;
  bool rt_callable_make_adjoint = false;
  bool rt_callable_make_controlled = false;

  bool rt_qubit_allocate = false;
  bool rt_qubit_allocate_array = false;
  bool rt_qubit_release = false;
  bool rt_qubit_release_array = false;

  bool rt_fail = false;
  bool rt_message = false;

  /// Big int and reference counting omitted for now.

  bool supports_output_recording() {
    return rt_tuple_record_output && rt_array_record_output && rt_result_record_output;
  }

  // ... and more composites.
};

class runtime_features {
public:
  runtime_features() = default;

  static runtime_features base_profile() {
    auto features = runtime_features();
    auto llvm = features.llvm_instructions = supported_llvm_instructions();
    llvm.call = true;
    llvm.br = true;
    llvm.ret = true;
    llvm.inttoptr = true;
    llvm.getelementptr = true;

    auto qir = features.qir_instructions = supported_qir_instructions();
    qir.rt_initialize = true;
    qir.rt_tuple_record_output = true; 
    qir.rt_array_record_output = true;
    qir.rt_result_record_output = true;

    return features;
  }

  static runtime_features adaptive_profile() {
    // ... set up with default adaptive profile settings.
  }

  supported_llvm_instructions llvm_instructions = supported_llvm_instructions();
  supported_qir_instructions qir_instructions = supported_qir_instructions();

  bool supports_base_profile() {
    return qir_instructions.rt_result_record_output 
      && qir_instructions.rt_initialize 
      && llvm_instructions.call
      && llvm_instructions.br
      && llvm_instructions.ret
      && llvm_instructions.inttoptr
      && llvm_instructions.getelementptr;
  }

  /// If this runtime can support dynamic arguments.
  bool supports_arguments = false;

  /// Supports classical returns. Means that the values returned from the entry-point are actually meaningful.
  bool supports_return_values = false;

  /// Can this system support internally-driven distributed execution.
  bool supports_distributed_execution = false;

  /// Can this runtime support things that are full programs, not just kernels. In this case it means 
  /// they have multiple separate quantum blocks, complicated internal logic and the recording methods become invalid.
  /// Quantum programs only return results, not a distribution.
  bool supports_quantum_programs = false;

  /// Uses CUDA internally or delegates blocks to CUDA Quantum for any blocks it sees in QIR.
  bool has_cuda_acceleration = false;
};

/// In-depth detail of what will happen internally to run the current execution. 
class execution_plan {
public:
  execution_plan() = default;

  /// @brief If this execution plan is available or not. In some cases the feature is simply not available, 
  /// in others it means the execution is too complicated for a plan to be fully generated without running. 
  bool is_available() {
    return false;
  }

  /// @brief Lists runtime features actively used in this particular plan.
  runtime_features used_features;
  
  virtual std::string pretty_print() { 
    return "Unavailable"; 
  }
};

/// TODO: Results should have both an actual return from the QIR method and the recorded distribution.
/// Right now template is just commented out since it needs to be virtual AND templated, 
/// which without doesn't work via the simple approach.

// template<typename T>
class runtime_results {
public:
  runtime_results() = default;

  std::unordered_map<std::string, std::size_t> distribution_result;

  // T result;

  /// @brief Squashes this result into a sample result, if possible,
  sample_result as_sample_result() {
    auto result = ExecutionResult(distribution_result);
    return sample_result(result);
  }
};

/// TODO: Stub for actual arguments to be fed to runtime.
class qir_argument { };

class runtime_platform {
public:
  const runtime_features features;

  runtime_platform(runtime_features features) : features(features) {}

  runtime_platform(runtime_platform&) = default;
  runtime_platform(runtime_platform&&) = default;
  ~runtime_platform() = default;

  virtual runtime_results execute(std::string qir, std::vector<qir_argument> &args, runtime_config &config);

  /// @brief If supported, allows for the runtime to return precisely what will be run if this QIR is passed to it. 
  /// Allows for much more nuanced decisions to be made before actually sending something for execution. Also useful for debugging. 
  virtual execution_plan fetch_execution_plan(std::string qir, std::vector<qir_argument> args, runtime_config &config);

protected:
  void run_cuda() {
    /// TODO: A runtimes internals will call into this with appropriate arguments - and get appropriate returns - if it has a CUDA block to run.
    /// What form CUDA would take in QIR is currently a questionmark, and I'm unfamiliar with the best arguments/outputs from it.
  }
};

/// Proxy QPU that holds an instance of the real runtime to just act as a CUDAQ-acceptable gateway to its capabilities.
class proxy_QPU : public QPU {
private:
  ExecutionContext *active_context;
  std::shared_ptr<runtime_platform> runtime;

public:
  proxy_QPU(std::shared_ptr<runtime_platform> &platform) : QPU() {
    runtime = platform;
  };

  void setExecutionContext(ExecutionContext *context) override {
    active_context = context;
  };

  void resetExecutionContext() override {
    active_context = nullptr;
  };
  
  /// @brief Transform the quake into QIR to pass to the runtime platform. 
  std::string extract_qir(
    const std::string &kernelName, void *kernelArgs) {

    auto contextPtr = initializeMLIR();
    MLIRContext &context = *contextPtr.get();

    // Get the quake representation of the kernel
    auto quakeCode = get_quake_by_name(kernelName);
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
    if (!func->hasAttr(cudaq::entryPointAttrName))`
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
            "Attempting to parse pipelines failed ("
             + errMsg + ").");
      if (failed(pm.run(moduleOpIn)))
        throw std::runtime_error("Runtime platform Quake lowering failed.");
    };

    if (!runtime->features.supports_arguments) {
      cudaq::info("Run Quake Synth.\n");
      PassManager pm(&context);
      pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, kernelArgs));
      if (failed(pm.run(moduleOp)))
        throw std::runtime_error("Couldn't fold arguments into QIR.");
    }

    // Run the config-specified pass pipeline
    std::string passPipelineConfig = "canonicalize";
    runPassPipeline(passPipelineConfig, moduleOp);

    // Get the code gen translation
        std::string codegenTranslation = "";
    auto translation = cudaq::getTranslation(codegenTranslation);

    // Apply user-specified codegen
    std::string codeStr;
    {
      llvm::raw_string_ostream outStr(codeStr);
      // if (disableMLIRthreading)
      //   moduleOp.getContext()->disableMultithreading();
      std::string postCodeGenPasses = "";
      bool printIR = false;
      bool enablePrintMLIREachPass = false;
      if (failed(translation(moduleOp, outStr, postCodeGenPasses, printIR,
                              enablePrintMLIREachPass)))
        throw std::runtime_error("Could not successfully translate to " +
                                  codegenTranslation + ".");
    }

    return codeStr;
  }

  /// @brief Extract QIR from this kernal and send it to execute against our attached 
  /// runtime.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {

    // Get the Quake code, lowered according to config file.
    auto qir = extract_qir(kernelName, args);

    // Get the current execution context and number of shots
    std::size_t localShots = 1000;
    if (executionContext->shots != std::numeric_limits<std::size_t>::max() &&
        executionContext->shots != 0)
      localShots = executionContext->shots;

    // TODO: Turn incoming arguments into runtime-appropriate ones. void-*-be-gone.
    auto runtime_args = std::vector<qir_argument>{};
    auto config = runtime_config(localShots);
    
    if (executionContext && executionContext->asyncExec) {
      executionContext->futureResult = cudaq::details::future(std::async(
        std::launch::async,
        [&, qir, args, config]() mutable -> cudaq::sample_result {
          auto results = runtime->execute(qir, runtime_args, config);
          return results.as_sample_result();
        }));
      return;  
    }

    auto results = runtime->execute(qir, runtime_args, config);
    if (executionContext) {
      executionContext->result = results.as_sample_result();
    }
  }
};

CUDAQ_REGISTER_TYPE(cudaq::QPU, proxy_QPU, proxy_qpu)

/// Platform instance that just proxies all meaningful requests to its singular QPU instance, which itself
/// is just a proxy to forward all information to the real runtime platform.
class proxy_platform : public quantum_platform {
private:
  std::shared_ptr<proxy_QPU> active_runtime;

public:
  proxy_platform() : quantum_platform() {
    // TODO: Decide which runtime to actually initialize with.
    auto platform = std::make_shared<runtime_platform>(runtime_platform(runtime_features()));
    platformQPUs.emplace_back(active_runtime = std::make_shared<proxy_QPU>(proxy_QPU(platform)));
  };

  /// @brief This setting is ignored because what QPU to run on is up to the runtime to decide.
  void set_current_qpu(const std::size_t device_id) override  {}

  std::size_t get_current_qpu() override { return 0; }

  // Task distribution is taken care of by the runtime itself, so we always just say no here.
  bool supports_task_distribution() const override { 
    return false; 
  }
};

} // namespace cudaq

CUDAQ_REGISTER_PLATFORM(proxy_platform, proxy_platform)