/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"

LLVM_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

namespace cudaq {

KernelThunkResultType QPU::launchModule(const std::string &name,
                                        mlir::ModuleOp module,
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
} // namespace cudaq
