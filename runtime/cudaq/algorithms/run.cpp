/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "run.h"
#include "common/RecordLogParser.h"

cudaq::detail::RunResultSpan cudaq::detail::convertToRunResultSpan(
    const std::string &outputLog,
    const cudaq_internal::compiler::LayoutInfoType &layoutInfo,
    std::size_t resultSize) {

  // 1. Pass the outputLog to the parser (target-specific?)
  cudaq::RecordLogParser parser(layoutInfo);
  parser.parse(outputLog);

  // 2. Get the buffer and length of buffer (in bytes) from the parser.
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  std::size_t resultCount = parser.getResultCount();

  // Validate that the buffer size is consistent with the successful shot count.
  // A result type without data, such as an empty struct, decodes to no bytes.
  if (bufferSize != 0 &&
      (bufferSize / resultSize != resultCount || bufferSize % resultSize != 0))
    throw std::runtime_error("run: " + std::to_string(resultCount) +
                             " decoded results occupy " +
                             std::to_string(bufferSize) +
                             " bytes, but the kernel's result type takes " +
                             std::to_string(resultSize) + " bytes.");
  char *buffer = nullptr;
  if (bufferSize != 0) {
    buffer = static_cast<char *>(::operator new(bufferSize));
    std::memcpy(buffer, origBuffer, bufferSize);
  }
  // The copy refers to the storage of each array, so it takes that storage
  // over from the parser.
  parser.releaseArrayStorage();

  // 3. Pass the span back as a RunResultSpan. NB: it is the responsibility of
  // the caller to free the buffer and the storage of each array it refers to.
  return {buffer, bufferSize, resultCount};
}
