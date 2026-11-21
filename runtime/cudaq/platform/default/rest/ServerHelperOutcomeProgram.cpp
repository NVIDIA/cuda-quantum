/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteRESTQPU.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/algorithms/sample_vnext.h"

namespace cudaq {
void RemoteRESTQPU::executeOutcomeProgram(const CompiledModule &program) {
  if (emulate)
    throw std::runtime_error(
        "ServerHelper outcome prototype requires direct REST execution");
  auto request = cudaq_internal::compiler::compileServerHelperFrequencies(
      program, getCompileTarget());
  if (request.shots == 0) {
    detail::publish_histogram({});
    return;
  }
  auto codes =
      cudaq_internal::compiler::CompiledModuleHelper::toKernelExecutions(
          request.circuit);
  executor->setShots(request.shots);
  // This tag selects the existing provider-count decoder. The requested
  // computation and shot count were extracted from the submitted program.
  auto histogram =
      executor->execute(codes, detail::ExecutionContextType::sample).get();
  detail::publish_histogram(std::move(histogram));
}
} // namespace cudaq
