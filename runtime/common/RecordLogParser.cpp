/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RecordLogParser.h"
#include "Logger.h"
#include "Timing.h"

void cudaq::RecordLogParser::parse(const std::string &outputLog) {
  ScopedTraceWithContext(cudaq::TIMING_RUN, "RecordLogParser::parse");
  cudaq::debug("Parsing log:\n{}", outputLog);
  std::vector<std::string> lines = cudaq::split(outputLog, '\n');
  if (lines.empty())
    return;
  for (const auto &line : lines) {
    std::vector<std::string> entries = cudaq::split(line, '\t');
    if (entries.empty())
      continue;
    handleRecord(entries);
  }
}

void cudaq::RecordLogParser::handleRecord(
    const std::vector<std::string> &entries) {
  const std::string &recordType = entries[0];
  if (recordType == "HEADER")
    handleHeader(entries);
  else if (recordType == "METADATA")
    handleMetadata(entries);
  else if (recordType == "START")
    handleStart(entries);
  else if (recordType == "OUTPUT")
    handleOutput(entries);
  else if (recordType == "END")
    handleEnd(entries);
  else
    throw std::runtime_error("Invalid record type: " + recordType);
}

void cudaq::RecordLogParser::handleHeader(
    const std::vector<std::string> &entries) {
  if (entries.size() < 3)
    throw std::runtime_error("Invalid HEADER record");
  if (entries[1] == "schema_name") {
    if (entries[2] == "labeled")
      schema = RecordSchemaType::LABELED;
    else if (entries[2] == "ordered")
      schema = RecordSchemaType::ORDERED;
    else
      throw std::runtime_error("Unknown schema type");
  }
  /// TODO: Handle schema version if needed
}

void cudaq::RecordLogParser::handleMetadata(
    const std::vector<std::string> &entries) {
  // Ignore metadata for now
}

void cudaq::RecordLogParser::handleStart(
    const std::vector<std::string> &entries) {
  // Ignore start of a shot for now
}

void cudaq::RecordLogParser::handleEnd(
    const std::vector<std::string> &entries) {
  if (entries.size() < 2)
    throw std::runtime_error("Missing shot status");
  if ("0" != entries[1])
    throw std::runtime_error("Cannot handle unsuccessful shot");
}

void cudaq::RecordLogParser::handleOutput(
    const std::vector<std::string> &entries) {
  if (entries.size() < 3)
    throw std::runtime_error("Insufficient data in a record");
  if ((schema == RecordSchemaType::LABELED) && (entries.size() != 4))
    throw std::runtime_error("Unexpected record size for a labeled record");
  const std::string &recType = entries[1];
  const std::string &recValue = entries[2];
  std::string recLabel = (entries.size() == 4) ? entries[3] : "";
  if (recType == "RESULT")
    throw std::runtime_error("This type is not yet supported");
  if (recType == "ARRAY") {
    containerMeta.m_type = ContainerType::ARRAY;
    containerMeta.elementCount = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = RecordSchemaType::LABELED;
      containerMeta.extractArrayInfo(recLabel);
      preallocateArray();
    }
    return;
  }
  if (recType == "TUPLE") {
    containerMeta.m_type = ContainerType::TUPLE;
    containerMeta.elementCount = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = RecordSchemaType::LABELED;
      containerMeta.extractTupleInfo(recLabel);
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
  if ((containerMeta.elementCount > 0) &&
      (schema == RecordSchemaType::LABELED)) {
    if (containerMeta.m_type == ContainerType::ARRAY)
      processArrayEntry(recValue, recLabel);
    else if (containerMeta.m_type == ContainerType::TUPLE)
      processTupleEntry(recValue, recLabel);
    containerMeta.processedElements++;
    if (containerMeta.processedElements == containerMeta.elementCount) {
      containerMeta.reset();
    }
  } else
    processSingleRecord(recValue, recLabel);
}

cudaq::details::DataHandlerBase &
cudaq::RecordLogParser::getDataHandler(const std::string &dataType) {
  // Static handlers for different data types
  static details::DataHandler<bool> boolHandler(
      std::make_unique<details::BooleanConverter>());
  static details::DataHandler<std::int8_t> i8Handler(
      std::make_unique<details::IntegerConverter<std::int8_t>>());
  static details::DataHandler<std::int16_t> i16Handler(
      std::make_unique<details::IntegerConverter<std::int16_t>>());
  static details::DataHandler<std::int32_t> i32Handler(
      std::make_unique<details::IntegerConverter<std::int32_t>>());
  static details::DataHandler<std::int64_t> i64Handler(
      std::make_unique<details::IntegerConverter<std::int64_t>>());
  static details::DataHandler<float> f32Handler(
      std::make_unique<details::FloatConverter<float>>());
  static details::DataHandler<double> f64Handler(
      std::make_unique<details::FloatConverter<double>>());
  // Map data type to the corresponding handler
  if (dataType == "i1")
    return boolHandler;
  else if (dataType == "i8")
    return i8Handler;
  else if (dataType == "i16")
    return i16Handler;
  else if (dataType == "i32")
    return i32Handler;
  else if (dataType == "i64")
    return i64Handler;
  else if (dataType == "f32")
    return f32Handler;
  else if (dataType == "f64")
    return f64Handler;
  throw std::runtime_error("Unsupported data type: " + dataType);
}

void cudaq::RecordLogParser::preallocateArray() {
  cudaq::details::DataHandlerBase &dh = getDataHandler(containerMeta.arrayType);
  containerMeta.dataOffset =
      dh.allocateArray(bufferHandler, containerMeta.elementCount);
}

void cudaq::RecordLogParser::preallocateTuple() {
  containerMeta.dataOffset = bufferHandler.getBufferSize();
  if (dataLayoutInfo.first == 0) {
    // Packed data allocation since alignment info is not provided
    for (auto ty : containerMeta.tupleTypes) {
      cudaq::details::DataHandlerBase &dh = getDataHandler(ty);
      containerMeta.tupleOffsets.push_back(dh.allocateTuple(bufferHandler));
    }
  } else {
    if (dataLayoutInfo.second.size() != containerMeta.tupleTypes.size())
      throw std::runtime_error("Tuple size mismatch in kernel and label.");
    // Directly allocate memory for the tuple, update offsets
    bufferHandler.resizeBuffer(dataLayoutInfo.first);
    containerMeta.tupleOffsets = dataLayoutInfo.second;
  }
}

void cudaq::RecordLogParser::processSingleRecord(const std::string &recValue,
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
  cudaq::details::DataHandlerBase &dh = getDataHandler(label);
  dh.addRecord(bufferHandler, recValue);
}

void cudaq::RecordLogParser::processArrayEntry(const std::string &recValue,
                                               const std::string &recLabel) {
  std::size_t index = containerMeta.extractIndex(recLabel);
  if (index >= containerMeta.elementCount)
    throw std::runtime_error("Array index out of bounds");
  cudaq::details::DataHandlerBase &dh = getDataHandler(containerMeta.arrayType);
  dh.insertIntoArray(bufferHandler, containerMeta.dataOffset, index, recValue);
}

void cudaq::RecordLogParser::processTupleEntry(const std::string &recValue,
                                               const std::string &recLabel) {
  std::size_t index = containerMeta.extractIndex(recLabel);
  if (index >= containerMeta.elementCount)
    throw std::runtime_error("Tuple index out of bounds");
  cudaq::details::DataHandlerBase &dh =
      getDataHandler(containerMeta.tupleTypes[index]);
  dh.insertIntoTuple(
      bufferHandler,
      containerMeta.dataOffset + containerMeta.tupleOffsets[index], recValue);
}
