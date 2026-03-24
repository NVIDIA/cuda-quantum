/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/cudaq_utils.h"
#include <cstddef>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace cudaq {

/// QIR output schema
enum struct RecordSchemaType { LABELED, ORDERED };
enum struct RecordType { HEADER, METADATA, OUTPUT, START, END };
enum struct OutputType { RESULT, BOOL, INT, DOUBLE };
enum struct ContainerType { NONE, ARRAY, TUPLE };

namespace details {

//===----------------------------------------------------------------------===//
// Type conversion infrastructure for string-to-value parsing
//===----------------------------------------------------------------------===//

/// Abstract base class for string to specific datatype conversion
template <typename T>
class TypeConverterBase {
public:
  virtual ~TypeConverterBase() = default;
  virtual T convert(const std::string &value) const = 0;
};

class BooleanConverter : public TypeConverterBase<bool> {
public:
  bool convert(const std::string &value) const override {
    if (value == "true" || value == "True" || value == "1")
      return true;
    if (value == "false" || value == "False" || value == "0")
      return false;
    throw std::runtime_error("Invalid boolean value");
  }
};

template <typename T>
class IntegerConverter : public TypeConverterBase<T> {
public:
  T convert(const std::string &value) const override {
    if constexpr (sizeof(T) <= 4)
      return static_cast<T>(std::stoi(value));
    else
      return static_cast<T>(std::stoll(value));
  }
};

template <typename T>
class FloatConverter : public TypeConverterBase<T> {
public:
  T convert(const std::string &value) const override {
    if constexpr (std::is_same_v<T, float>)
      return std::stof(value);
    else
      return std::stod(value);
  }
};

//===----------------------------------------------------------------------===//
// Buffer management for storing decoded data
//===----------------------------------------------------------------------===//

/// A helper class to manage the underlying memory buffer used by
/// 'RecordLogParser'. For container types, i.e. composite records, the buffer
/// acts as outer vector with pointers to inner buffers.
class BufferHandler {
public:
  BufferHandler() = default;
  ~BufferHandler() = default;
  /// Copying and assignment not permitted
  BufferHandler(BufferHandler &other) = delete;
  BufferHandler &operator=(BufferHandler &other) = delete;
  BufferHandler(const BufferHandler &) = delete;
  BufferHandler &operator=(const BufferHandler &) = delete;
  BufferHandler(BufferHandler &&) = default;
  BufferHandler &operator=(BufferHandler &&) = default;

  void *getBufferPtr() const {
    return reinterpret_cast<void *>(const_cast<char *>(buffer.data()));
  }

  std::size_t getBufferSize() const { return buffer.size(); }

  void resizeBuffer(std::size_t more) { buffer.resize(buffer.size() + more); }

  template <typename T>
  void addPrimitiveRecord(T value) {
    std::size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    std::memcpy(buffer.data() + position, &value, sizeof(T));
  }

  template <typename T>
  size_t allocateArrayRecord(size_t arrSize) {
    size_t vectorOffset = buffer.size();
    if constexpr (std::is_same_v<T, bool>) {
      auto *allocation = new std::vector<bool>(arrSize);
      if (!allocation)
        throw std::runtime_error("Memory allocation failed");
      auto byteLength = sizeof(*allocation);
      buffer.resize(vectorOffset + byteLength);
      std::memcpy(buffer.data() + vectorOffset, allocation, byteLength);
      return vectorOffset;
    }

    buffer.resize(vectorOffset + 3 * sizeof(T *));
    size_t byteLength = arrSize * sizeof(T);
    T *innerBuffer = static_cast<T *>(malloc(byteLength));
    if (!innerBuffer)
      throw std::runtime_error("Memory allocation failed");
    std::memset(innerBuffer, 0, byteLength);
    /// Initialize the three pointers of the inner vector
    T *startPtr = innerBuffer;
    T *end0Ptr = innerBuffer + arrSize;
    T *end1Ptr = end0Ptr;
    /// Store the pointers into the outer vector (buffer)
    T **ptrLoc = reinterpret_cast<T **>(buffer.data() + vectorOffset);
    ptrLoc[0] = startPtr;
    ptrLoc[1] = end0Ptr;
    ptrLoc[2] = end1Ptr;
    return vectorOffset;
  }

  template <typename T>
  void insertIntoArray(size_t offset, std::size_t index, T value) {
    if constexpr (std::is_same_v<T, bool>) {
      auto v = reinterpret_cast<std::vector<bool> *>(buffer.data() + offset);
      (*v)[index] = value;
    } else {
      T **ptrLoc = reinterpret_cast<T **>(buffer.data() + offset);
      ptrLoc[0][index] = value;
    }
  }

  /// NOTE: This is used only if data layout (alignment) is missing
  template <typename T>
  size_t allocateTupleRecord() {
    size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    return position;
  }

  template <typename T>
  void insertIntoTuple(size_t offset, T value) {
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
  }

private:
  std::vector<char> buffer;
};

//===----------------------------------------------------------------------===//
// Container metadata tracking for composite / aggregate types
//===----------------------------------------------------------------------===//

/// A helper structure to hold the current state of the container being
/// processed.
/// TODO: Handle nested containers.
class ContainerMetadata {
public:
  ContainerMetadata() = default;

  void reset() {
    m_type = ContainerType::NONE;
    elementCount = 0;
    processedElements = 0;
    arrayType.clear();
    dataOffset = 0;
    tupleTypes.clear();
    tupleOffsets.clear();
  }

  /// Parse string like "array<i32 x 4>"
  void extractArrayInfo(const std::string &label) {
    auto isArray = label.find("array");
    auto lessThan = label.find('<');
    auto greaterThan = label.find('>');
    auto x = label.find('x');
    if ((isArray == std::string::npos) || (lessThan == std::string::npos) ||
        (greaterThan == std::string::npos) || (x == std::string::npos))
      throw std::runtime_error("Array label missing keyword");
    if (elementCount != static_cast<size_t>(std::stoi(
                            label.substr(x + 2, greaterThan - x - 2))))
      throw std::runtime_error("Array size mismatch in value and label.");
    arrayType = label.substr(lessThan + 1, x - lessThan - 2);
  }

  /// Parse string like "tuple<i32, f64>"
  void extractTupleInfo(const std::string &label) {
    auto isTuple = label.find("tuple");
    auto lessThan = label.find('<');
    auto greaterThan = label.find('>');
    if ((isTuple == std::string::npos) || (lessThan == std::string::npos) ||
        (greaterThan == std::string::npos))
      throw std::runtime_error("Invalid tuple label");
    std::string types = label.substr(lessThan + 1, greaterThan - lessThan - 1);
    tupleTypes = cudaq::split(types, ',');
    if (elementCount != tupleTypes.size())
      throw std::runtime_error("Tuple size mismatch in value and label.");
    for (auto &ty : tupleTypes)
      ty.erase(std::remove(ty.begin(), ty.end(), ' '), ty.end());
  }

  /// Parse string like "[0]" for array index, and ".0" for tuple index.
  std::size_t extractIndex(const std::string &label) {
    if ((label[0] == '[') && (label[label.size() - 1] == ']'))
      return std::stoi(label.substr(1, label.size() - 2));
    if (label[0] == '.')
      return std::stoi(label.substr(1, label.size() - 1));
    throw std::runtime_error("Index not found in label");
  }

  ContainerType m_type = ContainerType::ARRAY;
  std::size_t elementCount = 0;
  std::size_t processedElements = 0;
  std::size_t dataOffset = 0;
  std::string arrayType;
  std::vector<std::string> tupleTypes;
  std::vector<std::size_t> tupleOffsets;
};

//===----------------------------------------------------------------------===//
// Type-specific data processing
//===----------------------------------------------------------------------===//

/// Abstract base class for data handling depending on the type of the data
class DataHandlerBase {
public:
  virtual ~DataHandlerBase() = default;
  virtual void addRecord(BufferHandler &bh, const std::string &value) = 0;
  virtual size_t allocateArray(BufferHandler &bh, std::size_t arrSize) = 0;
  virtual void insertIntoArray(BufferHandler &bh, std::size_t offset,
                               std::size_t index, const std::string &value) = 0;
  virtual size_t allocateTuple(BufferHandler &bh) = 0;
  virtual void insertIntoTuple(BufferHandler &bh, std::size_t offset,
                               const std::string &value) = 0;
};

template <typename T>
class DataHandler : public DataHandlerBase {
private:
  std::unique_ptr<details::TypeConverterBase<T>> converter;

public:
  DataHandler(std::unique_ptr<details::TypeConverterBase<T>> conv)
      : converter(std::move(conv)) {}
  void addRecord(BufferHandler &bh, const std::string &value) override {
    bh.addPrimitiveRecord<T>(converter->convert(value));
  }
  size_t allocateArray(BufferHandler &bh, std::size_t arrSize) override {
    return bh.allocateArrayRecord<T>(arrSize);
  }
  void insertIntoArray(BufferHandler &bh, std::size_t offset, std::size_t index,
                       const std::string &value) override {
    bh.insertIntoArray<T>(offset, index, converter->convert(value));
  }
  size_t allocateTuple(BufferHandler &bh) override {
    return bh.allocateTupleRecord<T>();
  }
  void insertIntoTuple(BufferHandler &bh, std::size_t offset,
                       const std::string &value) override {
    bh.insertIntoTuple<T>(offset, converter->convert(value));
  }
};

} // namespace details

namespace {
// Simplify look up of the required number of results by using a common
// identifier instead of different QIR versions (0.1 and 1.0)
constexpr char ResultCountMetadataName[] = "required_results";
} // namespace

//===----------------------------------------------------------------------===//
// Main record log parser and decoder class
//===----------------------------------------------------------------------===//

/// Simple decoder for translating QIR recorded results to a C++ binary data
/// structure.
class RecordLogParser {
public:
  RecordLogParser() = default;
  RecordLogParser(
      const std::pair<std::size_t, std::vector<std::size_t>> &layoutInfo)
      : dataLayoutInfo(layoutInfo) {}
  ~RecordLogParser() = default;

  /// Does the heavy-lifting of parsing the output log and converting it to a
  /// binary data structure that is compatible with the C++ host code. The data
  /// structure is created in a generic memory buffer. The buffer's address and
  /// length may be queried and returned as a result.
  void parse(const std::string &outputLog);

  /// Get a pointer to the data buffer. Note that the data buffer will be
  /// deallocated as soon as the RecordLogParser object is deconstructed.
  void *getBufferPtr() const { return bufferHandler.getBufferPtr(); }

  /// Get the size of the data buffer (in bytes).
  std::size_t getBufferSize() const { return bufferHandler.getBufferSize(); }

private:
  /// Process different types of records
  void handleHeader(const std::vector<std::string> &);
  void handleMetadata(const std::vector<std::string> &);
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
  /// appropriate type and store in the pre-allocated buffer
  void processArrayEntry(const std::string &, const std::string &);
  void processTupleEntry(const std::string &, const std::string &);
  /// Get data handler for the specified type
  details::DataHandlerBase &getDataHandler(const std::string &dataType);

  RecordSchemaType schema = RecordSchemaType::ORDERED;
  OutputType currentOutput;
  /// Manages the underlying buffer storage
  details::BufferHandler bufferHandler;
  /// Tracks container metadata during decoding
  details::ContainerMetadata containerMeta;
  /// Data layout information
  std::pair<std::optional<std::size_t>, std::vector<std::size_t>>
      dataLayoutInfo = {std::nullopt, {}};
  /// Metadata information extracted from the log
  std::unordered_map<std::string, std::string> metadata = {
      {ResultCountMetadataName, "1"}};
};
} // namespace cudaq
