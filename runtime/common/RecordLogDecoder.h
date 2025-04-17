/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/cudaq_utils.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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
  OutputType extractPrimitiveType(const std::string &label) {
    if ('i' == label[0]) {
      auto digits = std::stoi(label.substr(1));
      if (1 == digits)
        return OutputType::BOOL;
      return OutputType::INT;
    } else if ('f' == label[0]) {
      return OutputType::DOUBLE;
    }
    throw std::runtime_error("Unknown datatype in label");
  }

  std::size_t extractContainerIndex(const std::string &label) {
    if (('[' == label[0]) && (']' == label[label.size() - 1]))
      return std::stoi(label.substr(1, label.size() - 1));
    if ('.' == label[0])
      return std::stoi(label.substr(1, label.size() - 1));
    throw std::runtime_error("Index not found in label");
  }

  /// Parse string like "array<i32 x 4>"
  std::pair<std::size_t, std::string>
  extractArrayInfo(const std::string &label) {
    auto isArray = label.find("array");
    auto lessThan = label.find('<');
    auto greaterThan = label.find('>');
    auto x = label.find('x');
    if ((isArray == std::string::npos) || (lessThan == std::string::npos) ||
        (greaterThan == std::string::npos) || (x == std::string::npos))
      throw std::runtime_error("Array label missing keyword");
    std::size_t arrSize = std::stoi(label.substr(x + 2, greaterThan - x - 2));
    std::string arrType = label.substr(lessThan + 1, x - lessThan - 2);
    return std::make_pair(arrSize, arrType);
  }

  template <typename T>
  void addPrimitiveRecord(T value) {
    /// ASKME: Is this efficient?
    std::size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    std::memcpy(buffer.data() + position, &value, sizeof(T));
  }

  template <typename T>
  void allocateArrayRecord(size_t arrSize) {
    prevPosition = buffer.size();
    buffer.resize(prevPosition + (sizeof(T) * arrSize));
  }

  template <typename T>
  void addEntryToArray(std::size_t index, T value) {
    std::memcpy(buffer.data() + prevPosition + (index * sizeof(T)), &value,
                sizeof(T));
  }

  void processSingleRecord(const std::string &recValue,
                           const std::string &recLabel);

  void processArrayEntry(const std::string &recValue,
                         const std::string &recLabel);

  std::vector<char> buffer;
  SchemaType schema = SchemaType::ORDERED;
  RecordType currentRecord;
  OutputType currentOutput;
  ContainerType currentContainer;
  std::size_t containerSize;
  std::size_t prevPosition;
  std::string arrayType;
};
} // namespace cudaq
