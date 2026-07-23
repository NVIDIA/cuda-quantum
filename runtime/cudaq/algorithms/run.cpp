/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "run.h"
#include "common/RecordLogParser.h"
#include "common/Timing.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"

cudaq::detail::RunResultSpan cudaq::detail::convertToRunResultSpan(
    const std::string &outputLog,
    const cudaq_internal::compiler::LayoutInfoType &layoutInfo) {

  // 1. Pass the outputLog to the parser (target-specific?)
  cudaq::RecordLogParser parser(layoutInfo);
  parser.parse(outputLog);

  // 2. Get the buffer and length of buffer (in bytes) from the parser.
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  std::size_t resultCount = parser.getResultCount();

  // Validate that the buffer size is consistent with the successful shot count.
  if (resultCount > 0 && bufferSize % resultCount != 0)
    throw std::runtime_error(
        "run: the number of result bytes (" + std::to_string(bufferSize) +
        ") is not evenly divisible by the number of decoded results (" +
        std::to_string(resultCount) + ").");
  char *buffer = nullptr;
  if (bufferSize != 0) {
    buffer = static_cast<char *>(malloc(bufferSize));
    if (!buffer)
      throw std::runtime_error("run: result buffer allocation failed.");
    std::memcpy(buffer, origBuffer, bufferSize);
  }

  // 3. Pass the span back as a RunResultSpan. NB: it is the responsibility of
  // the caller to free the buffer.
  return {buffer, bufferSize, resultCount};
}
