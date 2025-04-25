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
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace cudaq {

/// QIR output schema
enum struct SchemaType { LABELED, ORDERED };
enum struct RecordType { HEADER, METADATA, OUTPUT, START, END };
enum struct OutputType { RESULT, BOOL, INT, DOUBLE };
enum struct ContainerType { NONE, ARRAY, TUPLE };

namespace details {

/// Abstract base class for string to specific datatype conversion
template <typename T>
class TypeConverterBase {
public:
  virtual ~TypeConverterBase() = default;
  virtual T convert(const std::string &value) const = 0;
};

class BooleanConverter : public TypeConverterBase<char> {
public:
  char convert(const std::string &value) const override {
    if (value == "true" || value == "1")
      return 1;
    if (value == "false" || value == "0")
      return 0;
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

/// A helper structure to hold the current state of the container being
/// processed.
/// TODO: Handle nested containers.
class ContainerHandler {
public:
  ContainerHandler() = default;

  void reset() {
    m_type = ContainerType::NONE;
    m_size = 0;
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
    if (m_size != static_cast<size_t>(
                      std::stoi(label.substr(x + 2, greaterThan - x - 2))))
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
    if (m_size != tupleTypes.size())
      throw std::runtime_error("Tuple size mismatch in value and label.");
    for (auto &ty : tupleTypes)
      ty.erase(std::remove(ty.begin(), ty.end(), ' '), ty.end());
  }

  /// Parse string like "[0]" for array index, and ".0" for tuple index
  std::size_t extractIndex(const std::string &label) {
    if ((label[0] == '[') && (label[label.size() - 1] == ']'))
      return std::stoi(label.substr(1, label.size() - 2));
    if (label[0] == '.')
      return std::stoi(label.substr(1, label.size() - 1));
    throw std::runtime_error("Index not found in label");
  }

  ContainerType m_type = ContainerType::ARRAY;
  std::size_t m_size = 0;
  std::size_t processedElements = 0;
  std::size_t dataOffset = 0;
  std::string arrayType;
  std::vector<std::string> tupleTypes;
  std::vector<std::size_t> tupleOffsets;
};

/// A helper class to manage the underlying memory buffer used by
/// 'RecordLogDecoder'. For container types, i.e. composite records, the buffer
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

  template <typename T>
  void addPrimitiveRecord(T value) {
    std::size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    std::memcpy(buffer.data() + position, &value, sizeof(T));
  }

  template <typename T>
  size_t allocateArrayRecord(size_t arrSize) {
    size_t vectorOffset = buffer.size();
    buffer.resize(vectorOffset + 3 * sizeof(T *));
    size_t byteLength = arrSize * sizeof(T);
    /// ASKME: How to properly free this memory?
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
    T **ptrLoc = reinterpret_cast<T **>(buffer.data() + offset);
    ptrLoc[0][index] = value;
  }

  /// TODO: Revisit tuple parsing.
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

/// Abstract base class for type-specific data handling.
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

/// Maps QIR type strings (e.g., "i32", "f64") to the appropriate data handling
/// instance.
class DataHandlerFactory {
public:
  static std::unique_ptr<DataHandlerBase>
  createDataHandler(const std::string &type) {
    if (type == "i1")
      return std::make_unique<DataHandler<char>>(
          std::make_unique<details::BooleanConverter>());
    if (type == "i8")
      return std::make_unique<DataHandler<std::int8_t>>(
          std::make_unique<details::IntegerConverter<std::int8_t>>());
    if (type == "i16")
      return std::make_unique<DataHandler<std::int16_t>>(
          std::make_unique<details::IntegerConverter<std::int16_t>>());
    if (type == "i32")
      return std::make_unique<DataHandler<std::int32_t>>(
          std::make_unique<details::IntegerConverter<std::int32_t>>());
    if (type == "i64")
      return std::make_unique<DataHandler<std::int64_t>>(
          std::make_unique<details::IntegerConverter<std::int64_t>>());
    if (type == "f32")
      return std::make_unique<DataHandler<float>>(
          std::make_unique<details::FloatConverter<float>>());
    if (type == "f64")
      return std::make_unique<DataHandler<double>>(
          std::make_unique<details::FloatConverter<double>>());

    throw std::runtime_error("Unsupported data type");
  }
};

} // namespace details
} // namespace cudaq
