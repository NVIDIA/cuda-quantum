/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "common/Timing.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstring>
#include <stdexcept>

using namespace cudaq_internal::compiler;
using namespace cudaq;

CUDAQ_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

// Bridge so the Python extension can register PythonLauncher into this DSO's
// registry. CUDA-Q Registry uses static inline Head/Tail, so each DSO that
// instantiates the template gets its own copy; launchModule runs in this DSO
// and reads the empty list. Registering via this function adds to our list.
extern "C" void cudaq_add_module_launcher_node(void *node_ptr) {
  using Node = cudaq::Registry<cudaq::ModuleLauncher>::node;
  cudaq::Registry<cudaq::ModuleLauncher>::add_node(
      static_cast<Node *>(node_ptr));
}

cudaq::KernelThunkResultType
cudaq::QPU::unifiedLaunchModule(const AnyModule &module, KernelArgs args) {
  if (std::holds_alternative<SourceModule>(module))
    throw std::runtime_error(
        "This QPU does not support launching uncompiled SourceModule kernels; "
        "subclasses must override unifiedLaunchModule.");

  const auto &compiled = std::get<CompiledModule>(module);
  return runJITCompiledModule(compiled, args);
}

cudaq::KernelThunkResultType
cudaq::QPU::runJITCompiledModule(const CompiledModule &compiled,
                                 KernelArgs args) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::runJITCompiledModule",
                         compiled.getName());
  auto rawArgs = args.getTypeErased().value_or(std::span<void *const>{});
  auto funcPtr = compiled.getJit()->getFn();
  const auto &resultInfo = compiled.getResultInfo();
  if (!compiled.isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    auto argsCreator = compiled.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    // If the kernel has a result, copy it from the packed buffer into
    // rawArgs.back() (where the caller expects to find it).
    if (resultInfo.hasResult()) {
      auto offset = compiled.getReturnOffset().value();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.getBufferSize());
    }
    std::free(buff);
    return {nullptr, 0};
  }
  if (resultInfo.hasResult()) {
    // Fully specialized with result: rawArgs.back() is the pre-allocated
    // result buffer; pass it directly to the thunk.
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  // Fully specialized, no result.
  funcPtr();
  return {nullptr, 0};
}

cudaq::CompiledModule cudaq::QPU::compileModule(const SourceModule &src,
                                                KernelArgs args,
                                                bool isEntryPoint) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `compileModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::compileModule",
                         src.getName());
  return launcher->compileModule(src, args, isEntryPoint);
}

void QPU::handleObservation(ExecutionContext &context) const {
  // The reason for the 2 if checks is simply to do a flushGateQueue() before
  // initiating the trace.
  bool execute = context.name == "observe";
  if (execute) {
    ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                           "handleObservation flushGateQueue()");
    getExecutionManager()->flushGateQueue();
  }
  if (execute) {
    ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                           "QPU::handleObservation (after flush)");
    double sum = 0.0;
    if (!context.spin.has_value())
      throw std::runtime_error("[QPU] Observe ExecutionContext specified "
                               "without a cudaq::spin_op.");

    std::vector<cudaq::ExecutionResult> results;
    cudaq::spin_op &H = context.spin.value();
    assert(cudaq::spin_op::canonicalize(H) == H);

    // If the backend supports the observe task, let it compute the
    // expectation value instead of manually looping over terms, applying
    // basis change ops, and computing <ZZ..ZZZ>
    if (context.canHandleObserve) {
      auto [exp, data] = cudaq::measure(H);
      context.expectationValue = exp;
      context.result = data;
    } else {

      // Loop over each term and compute coeff * <term>
      for (const auto &term : H) {
        if (term.is_identity())
          sum += term.evaluate_coefficient().real();
        else {
          // This takes a longer time for the first iteration unless
          // flushGateQueue() is called above.
          auto [exp, data] = cudaq::measure(term);
          results.emplace_back(data.to_map(), term.get_term_id(), exp);
          sum += term.evaluate_coefficient().real() * exp;
        }
      };

      context.expectationValue = sum;
      context.result = cudaq::sample_result(sum, results);
    }
  }
}
