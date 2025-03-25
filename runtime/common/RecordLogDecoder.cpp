/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RecordLogDecoder.h"

void cudaq::RecordLogDecoder::decode(const std::string &outputLog) {
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
    } break;
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
    } break;
    case RecordType::OUTPUT: {
      if (entries.size() < 3)
        throw std::runtime_error("Insufficent data in a record");
      if ((schema == SchemaType::LABELED) && (entries.size() != 4))
        throw std::runtime_error("Unexpected record size for a labeled record");

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

      processSingleRecord(recValue, recLabel);
    } break;
    }
  } // for line
}

void cudaq::RecordLogDecoder::processSingleRecord(const std::string &recValue,
                                                  const std::string &recLabel) {
  if ((!recLabel.empty()) && (extractPrimitiveType(recLabel) != currentOutput))
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
    addPrimitiveRecord<char>((char)value);
  } break;
  case OutputType::INT:
    if (recLabel == "i8")
      addPrimitiveRecord<std::int8_t>(std::stoi(recValue));
    else if (recLabel == "i16")
      addPrimitiveRecord<std::int16_t>(std::stoi(recValue));
    else if (recLabel == "i32")
      addPrimitiveRecord<std::int32_t>(std::stoi(recValue));
    else if (recLabel == "i64")
      addPrimitiveRecord<std::int64_t>(std::stoi(recValue));
    else
      throw std::runtime_error("integer size is not supported");
    break;
  case OutputType::DOUBLE:
    if (recLabel == "f32")
      addPrimitiveRecord<float>(std::stod(recValue));
    else if (recLabel == "f64")
      addPrimitiveRecord<double>(std::stod(recValue));
    else
      throw std::runtime_error("floating-point size is not supported");
    break;
  default:
    throw std::runtime_error("Unsupported output type");
  }
}
