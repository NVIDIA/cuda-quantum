/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/cudaq_utils.h"
#include <string>
#include <vector>

/// REVISIT: Memory management!
namespace cudaq {

// QIR output schema
enum struct SchemaType { LABELED, ORDERED };
enum struct RecordType { HEADER, METADATA, OUTPUT, START, END };
enum struct OutputType { RESULT, BOOL, INT, DOUBLE };
enum struct ContainerType { ARRAY, TUPLE };

// A container for the results from a single shot of executing a kernel which
// returns data of type `T`. The return value is stored in memory buffer
// starting with `buffer` and size in bytes in `size`.
/// TODO: Look at `RunResultSpan` type
struct OutputRecord {
  void *buffer;
  std::size_t size;
};

// Helper to keep track of the state of the parser logic.
struct ContainerIterator {
  ContainerType cTy;
  std::size_t cSize = 0;
  std::size_t cIndex = 0;
  std::size_t remaining = 0;

  void initialize(ContainerType c) {
    cTy = c;
    cSize = 0;
    cIndex = 0;
    remaining = 0;
  }

  bool isFull() { return ((cSize > 0) && (remaining == 0)); }
};

// A generic parser for QIR output record logs.
struct RecordParser {
private:
  SchemaType schema = SchemaType::ORDERED;
  RecordType currentRecord;
  OutputType currentOutput;
  bool isInContainer = false;
  ContainerIterator containerIt;

  template <typename T>
  void addPrimitiveRecord(T value) {
    results.emplace_back(
        OutputRecord{static_cast<void *>(new T(value)), sizeof(T)});
  }

  template <typename T>
  void addArrayRecord(size_t arrSize) {
    T *resArr = new T[arrSize];
    results.emplace_back(
        OutputRecord{static_cast<void *>(resArr), sizeof(T) * arrSize});
  }

  template <typename T>
  void addToArray(T value) {
    if (SchemaType::ORDERED == schema) {
      if (0 == containerIt.cIndex) {
        addArrayRecord<T>(containerIt.cSize);
        containerIt.remaining = containerIt.cSize;
      }
      static_cast<T *>(results.back().buffer)[containerIt.cIndex++] = value;
    } else if (SchemaType::LABELED == schema) {
      static_cast<T *>(results.back().buffer)[containerIt.cIndex] = value;
    }
    containerIt.remaining--;
    if (containerIt.isFull())
      isInContainer = false;
  }

  OutputType extractPrimitiveType(const std::string &label) {
    if ('i' == label[0]) {
      auto digits = std::stoi(label.substr(1));
      if (1 == digits)
        return OutputType::BOOL;
      return OutputType::INT;
    } else if ('f' == label[0])
      return OutputType::DOUBLE;
    throw std::runtime_error("Unknown datatype in label");
  }

  std::size_t extractContainerIndex(const std::string &label) {
    if (('[' == label[0]) && (']' == label[label.size() - 1]))
      return std::stoi(label.substr(1, label.size() - 1));
    if ('.' == label[0])
      return std::stoi(label.substr(1, label.size() - 1));
    throw std::runtime_error("Index not found in label");
  }

  /// Parse string like "array<3 x i32>"
  std::pair<std::size_t, OutputType>
  extractArrayInfo(const std::string &label) {
    auto isArray = label.find("array");
    auto lessThan = label.find('<');
    auto greaterThan = label.find('>');
    auto x = label.find('x');
    if ((isArray == std::string::npos) || (lessThan == std::string::npos) ||
        (greaterThan == std::string::npos) || (x == std::string::npos))
      throw std::runtime_error("Array label missing keyword");
    std::size_t arrSize =
        std::stoi(label.substr(lessThan + 1, x - lessThan - 2));
    OutputType arrType =
        extractPrimitiveType(label.substr(x + 2, greaterThan - x - 2));
    return std::make_pair(arrSize, arrType);
  }

  void prcoessSingleRecord(const std::string &recValue,
                           const std::string &recLabel) {
    if (!recLabel.empty()) {
      if (isInContainer)
        containerIt.cIndex = extractContainerIndex(recLabel);
      else if (extractPrimitiveType(recLabel) != currentOutput)
        throw std::runtime_error("Type mismatch in label");
    }
    switch (currentOutput) {
    case OutputType::BOOL: {
      bool value;
      if ("true" == recValue)
        value = true;
      else if ("false" == recValue)
        value = false;
      else
        throw std::runtime_error("Invalid boolean value");
      if (isInContainer)
        addToArray<bool>(value);
      else
        addPrimitiveRecord<bool>(value);
      break;
    }
    case OutputType::INT: {
      if (isInContainer)
        addToArray<int>(std::stoi(recValue));
      else
        addPrimitiveRecord<int>(std::stoi(recValue));
      break;
    }
    case OutputType::DOUBLE: {
      if (isInContainer)
        addToArray<double>(std::stod(recValue));
      else
        addPrimitiveRecord<double>(std::stod(recValue));
      break;
    }
    default:
      throw std::runtime_error("Unsupported output type");
    }
  }

public:
  std::vector<OutputRecord> results;

  std::vector<OutputRecord> parse(const std::string &data) {
    std::vector<std::string> lines = cudaq::split(data, '\n');
    if (lines.empty())
      return {};

    for (auto line : lines) {
      std::vector<std::string> entries = cudaq::split(line, '\t');
      if (entries.empty())
        continue;

      if ("HEADER" == entries[0])
        currentRecord = RecordType::HEADER;
      else if ("METADATA" == entries[0])
        currentRecord = RecordType::METADATA;
      else if ("OUTPUT" == entries[0])
        currentRecord = RecordType::OUTPUT;
      else if ("START" == entries[0])
        currentRecord = RecordType::START;
      else if ("END" == entries[0])
        currentRecord = RecordType::END;
      else
        throw std::runtime_error("Invalid data");

      switch (currentRecord) {
      case RecordType::HEADER: {
        if ("schema_name" == entries[1]) {
          if ("labeled" == entries[2])
            schema = SchemaType::LABELED;
          else if ("ordered" == entries[2])
            schema = SchemaType::ORDERED;
          else
            throw std::runtime_error("Unknown schema type");
        }
        /// TODO: Check schema version
        break;
      }
      case RecordType::METADATA:
        // ignore metadata for now
        break;
      case RecordType::START:
        // indicates start of a shot
        break;
      case RecordType::END: {
        // indicates end of a shot
        if (entries.size() < 2)
          throw std::runtime_error("Missing shot status");
        if ("0" != entries[1])
          throw std::runtime_error("Cannot handle unsuccessful shot");
        break;
      }
      case RecordType::OUTPUT: {
        if (entries.size() < 3)
          throw std::runtime_error("Insufficent data in a record");
        if ((schema == SchemaType::LABELED) && (entries.size() != 4))
          throw std::runtime_error(
              "Unexpected record size for a labeled record");

        std::string recType = entries[1];
        std::string recValue = entries[2];
        std::string recLabel = (entries.size() == 4) ? entries[3] : "";

        if ("RESULT" == recType)
          throw std::runtime_error("This type is not yet supported");
        if ("TUPLE" == recType)
          throw std::runtime_error("This type is not yet supported");

        if ("ARRAY" == recType) {
          isInContainer = true;
          containerIt.initialize(ContainerType::ARRAY);
          containerIt.cSize = std::stoi(recValue);
          if (0 == containerIt.cSize)
            throw std::runtime_error("Got empty array");
          if (SchemaType::LABELED == schema) {
            auto info = extractArrayInfo(recLabel);
            if (info.first != containerIt.cSize)
              throw std::runtime_error("Array size mismatch in label");
            containerIt.remaining = containerIt.cSize;
            if (OutputType::BOOL == info.second)
              addArrayRecord<bool>(info.first);
            else if (OutputType::INT == info.second)
              addArrayRecord<int>(info.first);
            else if (OutputType::DOUBLE == info.second)
              addArrayRecord<double>(info.first);
          }
        } else {
          if ("BOOL" == recType)
            currentOutput = OutputType::BOOL;
          else if ("INT" == recType)
            currentOutput = OutputType::INT;
          else if ("DOUBLE" == recType)
            currentOutput = OutputType::DOUBLE;
          else
            throw std::runtime_error("Invalid data");
          prcoessSingleRecord(recValue, recLabel);
        }
        break;
      }
      default:
        throw std::runtime_error("Unknown record type");
      }
    } // for line
    return results;
  }
};

} // namespace cudaq
