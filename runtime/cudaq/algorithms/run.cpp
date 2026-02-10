/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "run.h"
#include "common/LayoutInfo.h"
#include "common/RecordLogParser.h"
#include "common/Timing.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"

cudaq::details::RunResultSpan cudaq::details::runTheKernel(
    std::function<void()> &&kernel, quantum_platform &platform,
    const std::string &kernel_name, const std::string &original_name,
    std::size_t shots, const LayoutInfoType &layoutInfo, std::size_t qpu_id) {
  ScopedTraceWithContext(cudaq::TIMING_RUN, "runTheKernel");
  // 1. Clear the outputLog.
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  circuitSimulator->outputLog.clear();

  // Some platforms do not support run yet, emit error.
  if (!platform.get_codegen_config().outputLog)
    throw std::runtime_error("`run` is not yet supported on this target.");

  // 2. Launch the kernel on the QPU.
  if (platform.is_remote() || platform.is_emulated() ||
      platform.get_remote_capabilities().isRemoteSimulator) {
    // In a remote simulator execution or hardware emulation environment, set
    // the `run` context name and number of iterations (shots)
    cudaq::ExecutionContext ctx("run", shots, qpu_id);
    // Launch the kernel a single time to post the 'run' request to the remote
    // server or emulation executor.
    platform.with_execution_context(ctx, std::move(kernel));
    // Retrieve the result output log.
    // FIXME: this currently assumes all the shots are good.
    std::string remoteOutputLog(ctx.invocationResultBuffer.begin(),
                                ctx.invocationResultBuffer.end());
    circuitSimulator->outputLog.swap(remoteOutputLog);
  } else {
    cudaq::ExecutionContext ctx("run", 1, qpu_id);
    for (std::size_t i = 0; i < shots; ++i) {
      // Set the execution context since as noise model is attached to this
      // context.
      platform.with_execution_context(ctx, std::move(kernel));
    }
  }

  // 3. Pass the outputLog to the parser (target-specific?)
  cudaq::RecordLogParser parser(layoutInfo);
  parser.parse(circuitSimulator->outputLog);

  // 4. Get the buffer and length of buffer (in bytes) from the parser.
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);

  // 5. Clear the outputLog (?)
  circuitSimulator->outputLog.clear();

  // 6. Pass the span back as a RunResultSpan. NB: it is the responsibility of
  // the caller to free the buffer.
  return {buffer, bufferSize};
}
