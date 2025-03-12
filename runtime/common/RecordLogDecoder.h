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

namespace cudaq {

/// QIR output schema
enum struct SchemaType { LABELED, ORDERED };
enum struct RecordType { HEADER, METADATA, OUTPUT, START, END };
enum struct OutputType { RESULT, BOOL, INT, DOUBLE };
enum struct ContainerType { ARRAY, TUPLE };

/// Simple decoder for translating QIR recorded results to a C++ binary data
/// structure.
class RecordLogDecoder {

private:
  std::vector<char> buffer;
  SchemaType schema = SchemaType::ORDERED;
  RecordType currentRecord;
  OutputType currentOutput;

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

  void prcoessSingleRecord(const std::string &recValue,
                           const std::string &recLabel) {
    if ((!recLabel.empty()) &&
        (extractPrimitiveType(recLabel) != currentOutput))
      throw std::runtime_error("Type mismatch in label");

    switch (currentOutput) {
    case OutputType::BOOL: {
      bool value;
      if ("true" == recValue)
        value = true;
      else if ("false" == recValue)
        value = false;
      else
        throw std::runtime_error("Invalid boolean value");
      buffer.emplace_back(value ? '1' : '0');
      break;
    }
    // case OutputType::INT: {
    //   buffer.emplace_back(recValue.c_str());
    //   break;
    // }
    // case OutputType::DOUBLE: {
    //   addPrimitiveRecord<double>(std::stod(recValue));
    //   break;
    // }
    default:
      throw std::runtime_error("Unsupported output type");
    }
  }

public:
  RecordLogDecoder() = default;

  /// Does the heavy-lifting of parsing the output log and converting it to a
  /// binary data structure that is compatible with the C++ host code. The data
  /// structure is created in a generic memory buffer. The buffer's address and
  /// length may be queried and returned as a result.
  void decode(const std::string &outputLog) {
    std::vector<std::string> lines = cudaq::split(outputLog, '\n');
    if (lines.empty())
      return;

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
        if ("ARRAY" == recType)
          throw std::runtime_error("This type is not yet supported");

        if ("BOOL" == recType)
          currentOutput = OutputType::BOOL;
        else if ("INT" == recType)
          currentOutput = OutputType::INT;
        else if ("DOUBLE" == recType)
          currentOutput = OutputType::DOUBLE;
        else
          throw std::runtime_error("Invalid data");

        prcoessSingleRecord(recValue, recLabel);
        break;
      }
      default:
        throw std::runtime_error("Unknown record type");
      }
    } // for line
  }

  /// Get a pointer to the data buffer. Note that the data buffer will be
  /// deallocated as soon as the RecordLogDecoder object is deconstructed.
  void *getBufferPtr() const {
    return reinterpret_cast<void *>(const_cast<char *>(buffer.data()));
  }

  /// Get the size of the data buffer (in bytes).
  std::size_t getBufferSize() const { return buffer.size(); }
};
} // namespace cudaq
