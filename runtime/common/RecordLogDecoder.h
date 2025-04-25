/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "RecordLogDataUtils.h"
#include <unordered_map>

namespace cudaq {

/// Simple decoder for translating QIR recorded results to a C++ binary data
/// structure.
class RecordLogDecoder {
public:
  RecordLogDecoder() = default;
  ~RecordLogDecoder() = default;

  /// Does the heavy-lifting of parsing the output log and converting it to a
  /// binary data structure that is compatible with the C++ host code. The data
  /// structure is created in a generic memory buffer. The buffer's address and
  /// length may be queried and returned as a result.
  void decode(const std::string &outputLog);

  /// Get a pointer to the data buffer. Note that the data buffer will be
  /// deallocated as soon as the RecordLogDecoder object is deconstructed.
  void *getBufferPtr() const { return bufferHandler.getBufferPtr(); }

  /// Get the size of the data buffer (in bytes).
  std::size_t getBufferSize() const { return bufferHandler.getBufferSize(); }

private:
  /// Process different types of records
  void handleHeader(const std::vector<std::string> &);
  void handleMetadata(const std::vector<std::string> &);
  void handleStart(const std::vector<std::string> &);
  void handleEnd(const std::vector<std::string> &);
  /// Central dispatcher that handles different output types including scalar
  /// values, arrays, and tuples.
  void handleOutput(const std::vector<std::string> &);
  /// Allocate inner buffer for array records - one per shot
  void preallocateArray();
  /// Allocate contiguous memory for tuple records - one per shot
  void preallocateTuple();
  /// Process scalar values and non-labeled array/tuple entries
  void processSingleRecord(const std::string &, const std::string &);
  /// Extract index from label (out-of-order allowed), convert value to
  /// appropriate type and store in the preallocated buffer
  void processArrayEntry(const std::string &, const std::string &);
  void processTupleEntry(const std::string &, const std::string &);

  SchemaType schema = SchemaType::ORDERED;
  OutputType currentOutput;
  static const std::unordered_map<
      std::string,
      std::function<void(RecordLogDecoder *, const std::vector<std::string> &)>>
      recordHandlers;
  /// Manages the underlying buffer storage
  cudaq::details::BufferHandler bufferHandler;
  /// Tracks container metadata during decoding
  cudaq::details::ContainerHandler containerHandler;
  /// Cache of data handlers for different types
  std::unordered_map<std::string,
                     std::unique_ptr<cudaq::details::DataHandlerBase>>
      dataHandlerCache;
  cudaq::details::DataHandlerBase &getDataHandler(const std::string &dataType) {
    auto it = dataHandlerCache.find(dataType);
    if (it == dataHandlerCache.end()) {
      auto [newIt, _] = dataHandlerCache.emplace(
          dataType,
          cudaq::details::DataHandlerFactory::createDataHandler(dataType));
      return *newIt->second;
    }
    return *it->second;
  }
};
} // namespace cudaq
