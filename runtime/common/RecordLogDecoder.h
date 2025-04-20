/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/cudaq_utils.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudaq {

/// QIR output schema
enum struct SchemaType { LABELED, ORDERED };
enum struct RecordType { HEADER, METADATA, OUTPUT, START, END };
enum struct OutputType { RESULT, BOOL, INT, DOUBLE };
enum struct ContainerType { ARRAY, TUPLE };

/// Simple decoder for translating QIR recorded results to a C++ binary data
/// structure.
class RecordLogDecoder {
public:
  RecordLogDecoder() = default;

  /// Does the heavy-lifting of parsing the output log and converting it to a
  /// binary data structure that is compatible with the C++ host code. The data
  /// structure is created in a generic memory buffer. The buffer's address and
  /// length may be queried and returned as a result.
  void decode(const std::string &outputLog);

  /// Get a pointer to the data buffer. Note that the data buffer will be
  /// deallocated as soon as the RecordLogDecoder object is deconstructed.
  void *getBufferPtr() const {
    return reinterpret_cast<void *>(const_cast<char *>(buffer.data()));
  }

  /// Get the size of the data buffer (in bytes).
  std::size_t getBufferSize() const { return buffer.size(); }

private:
  /// A helper structure that provides handlers for various operations depending
  /// on the data types.
  struct TypeHandler {
    std::function<void(std::string)> addRecord;
    std::function<void(std::size_t)> allocateArray;
    std::function<void(std::size_t, std::string)> insertIntoArray;
    std::function<void()> allocateTuple;
    std::function<void(std::size_t, std::string)> insertIntoTuple;
  };

  /// A helper structure to hold the current state of the container being
  /// processed.
  /// TODO: Handle nested containers.
  struct ContainerHandler {
    ContainerType m_type = ContainerType::ARRAY;
    std::size_t m_size = 0;
    std::size_t processedElements = 0;
    std::size_t offset;
    std::string arrayType;
    std::vector<std::string> tupleTypes;
    std::vector<std::size_t> tupleOffsets;

    void reset();
    /// Parse string like "array<i32 x 4>"
    void extractArrayInfo(const std::string &);
    /// Parse string like "tuple<i32, f64>"
    void extractTupleInfo(const std::string &);
    /// Parse string like "[0]" for array index, and ".0" for tuple index
    std::size_t extractIndex(const std::string &);
  };

  void handleHeader(const std::vector<std::string> &);
  void handleMetadata(const std::vector<std::string> &);
  void handleStart(const std::vector<std::string> &);
  void handleEnd(const std::vector<std::string> &);
  void handleOutput(const std::vector<std::string> &);
  void preallocateArray();
  void preallocateTuple();
  bool convertToBool(const std::string &);
  void processSingleRecord(const std::string &, const std::string &);
  void processArrayEntry(const std::string &, const std::string &);
  void processTupleEntry(const std::string &, const std::string &);

  template <typename T>
  void addPrimitiveRecord(T value) {
    /// ASKME: Is this efficient?
    std::size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    std::memcpy(buffer.data() + position, &value, sizeof(T));
  }

  template <typename T>
  void allocateArrayRecord(size_t arrSize) {
    handler.offset = buffer.size();
    buffer.resize(handler.offset + (sizeof(T) * arrSize));
  }

  template <typename T>
  void insertIntoArray(std::size_t index, T value) {
    std::memcpy(buffer.data() + handler.offset + (index * sizeof(T)), &value,
                sizeof(T));
  }

  template <typename T>
  void insertIntoTuple(std::size_t index, T value) {
    std::memcpy(buffer.data() + handler.tupleOffsets[index], &value, sizeof(T));
  }

  std::vector<char> buffer;
  SchemaType schema = SchemaType::ORDERED;
  OutputType currentOutput;
  std::unordered_map<std::string, TypeHandler> dataTypeMap;
  ContainerHandler handler;
};
} // namespace cudaq
