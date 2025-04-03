/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "common/ExecutionContext.h"
#include "common/RecordLogDecoder.h"
#include "cudaq/simulators.h"
#include "nvqir/CircuitSimulator.h"

cudaq::details::RunResultSpan cudaq::details::runTheKernel(
    std::function<void()> &&kernel, quantum_platform &platform,
    const std::string &kernel_name, std::size_t shots) {
  // 1. Clear the outputLog.
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();

  circuitSimulator->outputLog.clear();

  // 2. Launch the kernel on the QPU.
  if (platform.get_remote_capabilities().isRemoteSimulator ||
      platform.is_emulated() || platform.is_remote()) {
    // In a remote simulator execution/hardware emulation environment, set the
    // `run` context name and number of iterations (shots)
    auto ctx = std::make_unique<cudaq::ExecutionContext>("run", shots);
    platform.set_exec_ctx(ctx.get());
    // Launch the kernel a single time to post the 'run' request to the remote
    // server or emulation executor.
    kernel();
    platform.reset_exec_ctx();
    // Retrieve the result output log.
    // FIXME: this currently assumes all the shots are good.
    std::string remoteOutputLog(ctx->invocationResultBuffer.begin(),
                                ctx->invocationResultBuffer.end());
    circuitSimulator->outputLog.swap(remoteOutputLog);
  } else {
    for (std::size_t i = 0; i < shots; ++i)
      kernel();
  }

  // 3. Pass the outputLog to the decoder (target-specific?)
  cudaq::RecordLogDecoder decoder;
  decoder.decode(circuitSimulator->outputLog);

  // 4. Get the buffer and length of buffer (in bytes) from the decoder.
  auto *origBuffer = decoder.getBufferPtr();
  std::size_t bufferSize = decoder.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);

  // 5. Clear the outputLog (?)
  circuitSimulator->outputLog.clear();

  // 6. Pass the span back as a RunResultSpan. NB: it is the responsibility of
  // the caller to free the buffer.
  return {buffer, bufferSize};
}
