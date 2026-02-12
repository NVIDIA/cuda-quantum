/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "mlir/IR/BuiltinOps.h"

LLVM_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

// Bridge so the Python extension can register PythonLauncher into this DSO's
// registry. LLVM's Registry uses static inline Head/Tail, so each DSO that
// instantiates the template gets its own copy; launchModule runs in this DSO
// and reads the empty list. Registering via this function adds to our list.
extern "C" void cudaq_add_module_launcher_node(void *node_ptr) {
  using Node = llvm::Registry<cudaq::ModuleLauncher>::node;
  llvm::Registry<cudaq::ModuleLauncher>::add_node(
      static_cast<Node *>(node_ptr));
}

cudaq::KernelThunkResultType
cudaq::QPU::launchModule(const std::string &name, mlir::ModuleOp module,
                         const std::vector<void *> &rawArgs,
                         mlir::Type resultTy) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `launchModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule", name);
  return launcher->launchModule(name, module, rawArgs, resultTy);
}

void *
cudaq::QPU::specializeModule(const std::string &name, mlir::ModuleOp module,
                             const std::vector<void *> &rawArgs,
                             mlir::Type resultTy,
                             std::optional<cudaq::JitEngine> &cachedEngine) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `specializeModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::specializeModule", name);
  return launcher->specializeModule(name, module, rawArgs, resultTy,
                                    cachedEngine);
}
