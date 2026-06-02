/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QPU.h"
#include "common/ArgumentWrapper.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/RuntimeTarget.h"
#include "common/Timing.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Verifier/QIRLLVMIRDialect.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h>

using namespace mlir;

/// Lowers \p module to LLVM code. The LLVM code will use "full QIR" as the
/// transport layer. If \p kernelName and \p args are provided, they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_qir_llvm(const std::string &name,
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args,
                                             const std::string &format) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);

  auto target = getDefaultPythonCompileTarget(other_policies{},
                                              cudaq::getExecutionContext());
  target->fullySpecialize = true;
  cudaq_internal::compiler::Compiler compiler(std::move(target));

  auto rawArgs = args.getArgs();
  auto compiled = compiler.runPassPipeline(name, module.getAsOpaquePointer(),
                                           {rawArgs}, true);
  auto compiled_module =
      cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
          *compiled.getMlir());

  // TODO: can the following be merged with qirProfileTranslationFunction in
  // RuntimeMLIR.cpp?
  PassManager pm(compiled_module.getContext());
  if (!compiler.getTarget().pipelineConfig.skipTargetLoweringPipeline) {
    cudaq::opt::addAggressiveInlining(pm);
    cudaq::opt::createTargetFinalizePipeline(pm);
  }
  cudaq::opt::addAOTPipelineConvertToQIR(pm, format);
  if (failed(cudaq_internal::compiler::runPassManager(pm, compiled_module)))
    throw std::runtime_error("Pass pipeline failed.");
  if (failed(cudaq::verifier::checkQIRLLVMIRDialect(compiled_module, format)))
    throw std::runtime_error("QIR conformance failed.");
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(compiled_module, llvmContext);
  if (!llvmModule)
    return "{translation failed}";
  std::string result;
  llvm::raw_string_ostream os(result);
  llvmModule->print(os, nullptr);
  os.flush();
  return result;
}

/// Lowers \p module to `Open QASM 2`. The output will be a string of `Open
/// QASM` code. \p kernelName and \p args should be provided, as they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_openqasm(const std::string &name,
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getASM", name);

  auto target = getDefaultPythonCompileTarget(other_policies{},
                                              cudaq::getExecutionContext());
  target->fullySpecialize = true;
  cudaq_internal::compiler::Compiler compiler(std::move(target));

  auto rawArgs = args.getArgs();
  auto compiled = compiler.runPassPipeline(name, module.getAsOpaquePointer(),
                                           {rawArgs}, true);
  auto compiled_module =
      cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
          *compiled.getMlir());
  auto *ctx = compiled_module.getContext();
  PassManager pm(ctx);
  if (!compiler.getTarget().pipelineConfig.skipTargetLoweringPipeline) {
    cudaq::opt::addAggressiveInlining(pm);
    cudaq::opt::createTargetFinalizePipeline(pm);
  }
  cudaq::opt::createPipelineTransformsForPythonToOpenQASM(pm);
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);
  const bool enablePrintMLIRBeforeAndAfterEachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  if (enablePrintMLIRBeforeAndAfterEachPass) {
    ctx->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(cudaq_internal::compiler::runPassManager(pm, compiled_module)))
    throw std::runtime_error("Pass pipeline failed.");
  std::string result;
  llvm::raw_string_ostream os(result);
  if (mlir::failed(cudaq::translateToOpenQASM(compiled_module, os)))
    return "{translation failed}";
  os.flush();
  return result;
}
