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

protected:
  /// Helper to process an output line.
  // Subclass can override to handle different output record format.
  virtual void processOutputField(const std::vector<std::string> &entries);
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

  template <typename T>
  void addPrimitiveRecord(T value) {
    /// ASKME: Is this efficient?
    std::size_t position = buffer.size();
    buffer.resize(position + sizeof(T));
    std::memcpy(buffer.data() + position, &value, sizeof(T));
  }

  void processSingleRecord(const std::string &recValue,
                           const std::string &recLabel);

  std::vector<char> buffer;
  SchemaType schema = SchemaType::ORDERED;
  RecordType currentRecord;
  OutputType currentOutput;
  /// Key name for an output record.
  std::string OutputFieldKey = "OUTPUT";
  bool hasShotStatus = true;
};

// Decoder for the legacy/tagged (unspec'ed) output format.
class TaggedRecordLogDecoder : public RecordLogDecoder {
public:
  TaggedRecordLogDecoder(const std::vector<std::string> &measureRegLabels = {})
      : resultRecordLabels(measureRegLabels) {
    OutputFieldKey = "RESULT";
    hasShotStatus = false;
  }
  virtual void
  processOutputField(const std::vector<std::string> &entries) override;

private:
  // These result record labels are ignored when parsing the output log as they
  // are not associated with the return value.
  // Note: any `mz` results shall be converted to bool upon return.
  // These output labels can be retrieved from the QIR that we send to the
  // backend. Hence, we use it to filter the returned log.
  std::vector<std::string> resultRecordLabels;
};
} // namespace cudaq
