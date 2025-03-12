/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>
#include <vector>

namespace cudaq {
/// Simple decoder for translating QIR recorded results to a C++ binary data
/// structure.
class RecordLogDecoder {
public:
  RecordLogDecoder() = default;

  /// Does the heavy-lifting of parsing the output log and converting it to a
  /// binary data structure that is compatible with the C++ host code. The data
  /// structure is created in a generic memory buffer. The buffer's address and
  /// length may be queried and returned as a result.
  void decode(const std::string &outputLog) {
    // NYI
  }

  /// Get a pointer to the data buffer. Note that the data buffer will be
  /// deallocated as soon as the RecordLogDecoder object is deconstructed.
  void *getBufferPtr() const {
    return reinterpret_cast<void *>(const_cast<char *>(buffer.data()));
  }

  /// Get the size of the data buffer (in bytes).
  std::size_t getBufferSize() const { return buffer.size(); }

private:
  std::vector<char> buffer;
};
} // namespace cudaq
