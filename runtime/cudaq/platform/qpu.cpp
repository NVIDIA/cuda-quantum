/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "mlir/IR/BuiltinOps.h"

using namespace cudaq_internal::compiler;

LLVM_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

cudaq::KernelThunkResultType
cudaq::QPU::launchModule(const std::string &name, mlir::ModuleOp module,
                         const std::vector<void *> &rawArgs) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `launchModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule", name);
  auto compiled = launcher->compileModule(name, module, rawArgs, true);
  return compiled.execute(rawArgs);
}

cudaq::CompiledKernel
cudaq::QPU::specializeModule(const std::string &name, mlir::ModuleOp module,
                             const std::vector<void *> &rawArgs,
                             bool isEntryPoint) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `specializeModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::specializeModule", name);
  return launcher->compileModule(name, module, rawArgs, isEntryPoint);
}
