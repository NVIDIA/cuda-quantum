/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "algorithms/observe/policy.h"
#include "algorithms/policies.h"
#include "algorithms/sample/policy.h"
#include "common/CompiledModule.h"
#include "common/KernelArgs.h"
#include "common/Timing.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstring>
#include <stdexcept>

using namespace cudaq_internal::compiler;
using namespace cudaq;

cudaq::KernelThunkResultType
cudaq::QPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the other_policies.");
}

sample_result cudaq::QPU::launchKernel(const sample_policy &policy,
                                       const CompiledModule &module,
                                       KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the sample_policy.");
}

async_sample_result cudaq::QPU::launchKernel(const async_sample_policy &policy,
                                             const CompiledModule &module,
                                             KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the async_sample_policy.");
}

orca::sample_policy::result_type
cudaq::QPU::launchKernel(const orca::sample_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the orca::sample_policy.");
}

orca::async_sample_policy::result_type
cudaq::QPU::launchKernel(const orca::async_sample_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the orca::async_sample_policy.");
}

observe_result cudaq::QPU::launchKernel(const observe_policy &policy,
                                        const CompiledModule &module,
                                        KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the observe_policy.");
}

run_result cudaq::QPU::launchKernel(const run_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the run_policy.");
}

async_run_policy::result_type
cudaq::QPU::launchKernel(const async_run_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the async_run_policy.");
}

msm_dimensions cudaq::QPU::launchKernel(const msm_size_policy &policy,
                                        const CompiledModule &module,
                                        KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the msm_size_policy.");
}

msm_result cudaq::QPU::launchKernel(const msm_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the msm_policy.");
}

async_observe_result
cudaq::QPU::launchKernel(const async_observe_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the async_observe_policy.");
}

dem_result cudaq::QPU::launchKernel(const dem_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the dem_policy.");
}

estimate_result cudaq::QPU::launchKernel(const estimate_policy &policy,
                                         const CompiledModule &module,
                                         KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the estimate_policy.");
}

ptsbe::sample_policy::result_type
cudaq::QPU::launchKernel(const ptsbe::sample_policy &policy,
                         const CompiledModule &module, KernelArgs args) {
  throw std::runtime_error(
      "This QPU does not support launching the ptsbe::sample_policy.");
}

cudaq::CompileTarget
cudaq::QPU::getCompileTarget(bool skipPipelineSubstitutions) {
  throw std::runtime_error("No CompileTarget defined for this QPU");
}
