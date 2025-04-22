/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RecordLogDecoder.h"

const std::unordered_map<std::string,
                         std::function<void(cudaq::RecordLogDecoder *,
                                            const std::vector<std::string> &)>>
    cudaq::RecordLogDecoder::recordHandlers = {
        {"HEADER",
         [](RecordLogDecoder *self, const std::vector<std::string> &entries) {
           self->handleHeader(entries);
         }},
        {"METADATA",
         [](RecordLogDecoder *self, const std::vector<std::string> &entries) {
           self->handleMetadata(entries);
         }},
        {"OUTPUT",
         [](RecordLogDecoder *self, const std::vector<std::string> &entries) {
           self->handleOutput(entries);
         }},
        {"START",
         [](RecordLogDecoder *self, const std::vector<std::string> &entries) {
           self->handleStart(entries);
         }},
        {"END",
         [](RecordLogDecoder *self, const std::vector<std::string> &entries) {
           self->handleEnd(entries);
         }}};

const std::unordered_map<std::string, cudaq::RecordLogDecoder::TypeHandler>
    cudaq::RecordLogDecoder::dataTypeMap = {
        {"i1",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<char>(
                static_cast<char>(self->convertToBool(value)));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<char>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<char>(
                index, static_cast<char>(self->convertToBool(value)));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(char));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(char);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<char>(
                index, static_cast<char>(self->convertToBool(value)));
          }}},
        {"i8",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<std::int8_t>(std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<std::int8_t>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<std::int8_t>(index, std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(std::int8_t));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(std::int8_t);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<std::int8_t>(index, std::stoi(value));
          }}},
        {"i16",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<std::int16_t>(std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<std::int16_t>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<std::int16_t>(index, std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(std::int16_t));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(std::int16_t);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<std::int16_t>(index, std::stoi(value));
          }}},
        {"i32",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<std::int32_t>(std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<std::int32_t>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<std::int32_t>(index, std::stoi(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(std::int32_t));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(std::int32_t);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<std::int32_t>(index, std::stoi(value));
          }}},
        {"i64",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<std::int64_t>(std::stoll(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<std::int64_t>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<std::int64_t>(index, std::stoll(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(std::int64_t));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(std::int64_t);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<std::int64_t>(index, std::stoll(value));
          }}},
        {"f32",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<float>(std::stof(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<float>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<float>(index, std::stof(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(float));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(float);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<float>(index, std::stof(value));
          }}},
        {"f64",
         {[](RecordLogDecoder *self, const std::string &value) {
            self->addPrimitiveRecord<double>(std::stod(value));
          },
          [](RecordLogDecoder *self) {
            self->allocateArrayRecord<double>(self->handler.m_size);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoArray<double>(index, std::stod(value));
          },
          [](RecordLogDecoder *self) {
            self->buffer.resize(self->handler.offset + sizeof(double));
            self->handler.tupleOffsets.push_back(self->handler.offset);
            self->handler.offset += sizeof(double);
          },
          [](RecordLogDecoder *self, std::size_t index,
             const std::string &value) {
            self->insertIntoTuple<double>(index, std::stod(value));
          }}}};

void cudaq::RecordLogDecoder::decode(const std::string &outputLog) {
  std::vector<std::string> lines = cudaq::split(outputLog, '\n');
  if (lines.empty())
    return;
  for (const auto &line : lines) {
    std::vector<std::string> entries = cudaq::split(line, '\t');
    if (entries.empty())
      continue;
    auto it = recordHandlers.find(entries[0]);
    if (it != recordHandlers.end()) {
      it->second(this, entries);
    } else {
      throw std::runtime_error("Invalid record type: " + entries[0]);
    }
  }
}

void cudaq::RecordLogDecoder::ContainerHandler::reset() {
  m_type = ContainerType::ARRAY;
  m_size = 0;
  processedElements = 0;
  offset = 0;
  arrayType.clear();
  tupleTypes.clear();
  tupleOffsets.clear();
}

void cudaq::RecordLogDecoder::ContainerHandler::extractArrayInfo(
    const std::string &label) {
  auto isArray = label.find("array");
  auto lessThan = label.find('<');
  auto greaterThan = label.find('>');
  auto x = label.find('x');
  if ((isArray == std::string::npos) || (lessThan == std::string::npos) ||
      (greaterThan == std::string::npos) || (x == std::string::npos))
    throw std::runtime_error("Array label missing keyword");
  if (m_size !=
      static_cast<size_t>(std::stoi(label.substr(x + 2, greaterThan - x - 2))))
    throw std::runtime_error("Array size mismatch in value and label.");
  arrayType = label.substr(lessThan + 1, x - lessThan - 2);
}

void cudaq::RecordLogDecoder::ContainerHandler::extractTupleInfo(
    const std::string &label) {
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

std::size_t cudaq::RecordLogDecoder::ContainerHandler::extractIndex(
    const std::string &label) {
  if ((label[0] == '[') && (label[label.size() - 1] == ']'))
    return std::stoi(label.substr(1, label.size() - 2));
  if (label[0] == '.')
    return std::stoi(label.substr(1, label.size() - 1));
  throw std::runtime_error("Index not found in label");
}

void cudaq::RecordLogDecoder::handleHeader(
    const std::vector<std::string> &entries) {
  if (entries.size() < 3)
    throw std::runtime_error("Invalid HEADER record");
  if (entries[1] == "schema_name") {
    if (entries[2] == "labeled")
      schema = SchemaType::LABELED;
    else if (entries[2] == "ordered")
      schema = SchemaType::ORDERED;
    else
      throw std::runtime_error("Unknown schema type");
  }
  /// TODO: Handle schema version if needed
}

void cudaq::RecordLogDecoder::handleMetadata(
    const std::vector<std::string> &entries) {
  // Ignore metadata for now
}

void cudaq::RecordLogDecoder::handleStart(
    const std::vector<std::string> &entries) {
  // Ignore start of a shot for now
}

void cudaq::RecordLogDecoder::handleEnd(
    const std::vector<std::string> &entries) {
  if (entries.size() < 2)
    throw std::runtime_error("Missing shot status");
  if ("0" != entries[1])
    throw std::runtime_error("Cannot handle unsuccessful shot");
}

void cudaq::RecordLogDecoder::handleOutput(
    const std::vector<std::string> &entries) {
  if (entries.size() < 3)
    throw std::runtime_error("Insufficient data in a record");
  if ((schema == SchemaType::LABELED) && (entries.size() != 4))
    throw std::runtime_error("Unexpected record size for a labeled record");
  const std::string &recType = entries[1];
  const std::string &recValue = entries[2];
  std::string recLabel = (entries.size() == 4) ? entries[3] : "";
  if (recType == "RESULT")
    throw std::runtime_error("This type is not yet supported");
  if (recType == "ARRAY") {
    handler.m_type = ContainerType::ARRAY;
    handler.m_size = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = SchemaType::LABELED;
      handler.extractArrayInfo(recLabel);
      preallocateArray();
    }
    return;
  }
  if (recType == "TUPLE") {
    handler.m_type = ContainerType::TUPLE;
    handler.m_size = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = SchemaType::LABELED;
      handler.extractTupleInfo(recLabel);
      preallocateTuple();
    }
    return;
  }
  if (recType == "BOOL")
    currentOutput = OutputType::BOOL;
  else if (recType == "INT")
    currentOutput = OutputType::INT;
  else if (recType == "DOUBLE")
    currentOutput = OutputType::DOUBLE;
  else
    throw std::runtime_error("Invalid data");
  if ((handler.m_size > 0) && (schema == SchemaType::LABELED)) {
    if (handler.m_type == ContainerType::ARRAY)
      processArrayEntry(recValue, recLabel);
    else if (handler.m_type == ContainerType::TUPLE)
      processTupleEntry(recValue, recLabel);
    handler.processedElements++;
    if (handler.processedElements == handler.m_size) {
      handler.reset();
    }
  } else
    processSingleRecord(recValue, recLabel);
}

void cudaq::RecordLogDecoder::preallocateArray() {
  auto it = dataTypeMap.find(handler.arrayType);
  if (it != dataTypeMap.end())
    it->second.allocateArray(this);
  else
    throw std::runtime_error("Unsupported array type");
}

void cudaq::RecordLogDecoder::preallocateTuple() {
  handler.offset = buffer.size();
  for (auto ty : handler.tupleTypes) {
    auto it = dataTypeMap.find(ty);
    if (it != dataTypeMap.end())
      it->second.allocateTuple(this);
    else
      throw std::runtime_error("Unsupported array type");
  }
}

bool cudaq::RecordLogDecoder::convertToBool(const std::string &value) {
  if ((value == "true") || (value == "1"))
    return true;
  else if ((value == "false") || (value == "0"))
    return false;
  else
    throw std::runtime_error("Invalid boolean value");
}

void cudaq::RecordLogDecoder::processSingleRecord(const std::string &recValue,
                                                  const std::string &recLabel) {
  auto label = recLabel;
  if (label.empty()) {
    if (currentOutput == OutputType::BOOL)
      label = "i1";
    else if (currentOutput == OutputType::INT)
      label = "i32";
    else if (currentOutput == OutputType::DOUBLE)
      label = "f64";
  }
  auto it = dataTypeMap.find(label);
  if (it != dataTypeMap.end())
    it->second.addRecord(this, recValue);
  else
    throw std::runtime_error("Unsupported output type");
}

void cudaq::RecordLogDecoder::processArrayEntry(const std::string &recValue,
                                                const std::string &recLabel) {
  std::size_t index = handler.extractIndex(recLabel);
  auto it = dataTypeMap.find(handler.arrayType);
  if (it != dataTypeMap.end())
    it->second.insertIntoArray(this, index, recValue);
  else
    throw std::runtime_error("Unsupported output type");
}

void cudaq::RecordLogDecoder::processTupleEntry(const std::string &recValue,
                                                const std::string &recLabel) {
  std::size_t index = handler.extractIndex(recLabel);
  auto it = dataTypeMap.find(handler.tupleTypes[index]);
  if (it != dataTypeMap.end())
    it->second.insertIntoTuple(this, index, recValue);
  else
    throw std::runtime_error("Unsupported tuple type");
}
