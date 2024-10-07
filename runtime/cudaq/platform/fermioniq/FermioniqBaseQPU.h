/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

class FermioniqBaseQPU : public BaseRemoteRESTQPU {
public:
  virtual bool isRemote() override { return true; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return false; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  void setNoiseModel(const cudaq::noise_model *model) override {
    throw std::runtime_error("Noise modeling is not allowed on this backend");
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

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::draw().");

    // Get the Quake code, lowered according to config file.
    auto codes = getQuakeCodes(kernelName, nullptr, rawArgs);
    completeLaunchKernel(kernelName, std::move(codes));
  }

  /// @brief Extract the Quake representation for the given kernel name and
  /// lower it to the code format required for the specific backend. The
  /// lowering process is controllable via the configuration file in the
  /// platform directory for the targeted backend.
  std::vector<cudaq::KernelExecution>
  getQuakeCodes(const std::string &kernelName, void *kernelArgs,
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
};
} // namespace cudaq
