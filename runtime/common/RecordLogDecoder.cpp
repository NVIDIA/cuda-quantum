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
  for (const auto &line : lines) {
    std::vector<std::string> entries = cudaq::split(line, '\t');
    if (entries.empty())
      continue;
    handleRecord(entries);
  }
}

void cudaq::RecordLogDecoder::handleRecord(
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
    containerHandler.m_type = ContainerType::ARRAY;
    containerHandler.m_size = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = SchemaType::LABELED;
      containerHandler.extractArrayInfo(recLabel);
      preallocateArray();
    }
    return;
  }
  if (recType == "TUPLE") {
    containerHandler.m_type = ContainerType::TUPLE;
    containerHandler.m_size = std::stoul(recValue);
    if (!recLabel.empty()) {
      schema = SchemaType::LABELED;
      containerHandler.extractTupleInfo(recLabel);
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
  if ((containerHandler.m_size > 0) && (schema == SchemaType::LABELED)) {
    if (containerHandler.m_type == ContainerType::ARRAY)
      processArrayEntry(recValue, recLabel);
    else if (containerHandler.m_type == ContainerType::TUPLE)
      processTupleEntry(recValue, recLabel);
    containerHandler.processedElements++;
    if (containerHandler.processedElements == containerHandler.m_size) {
      containerHandler.reset();
    }
  } else
    processSingleRecord(recValue, recLabel);
}

void cudaq::RecordLogDecoder::preallocateArray() {
  cudaq::details::DataHandlerBase &dh =
      getDataHandler(containerHandler.arrayType);
  containerHandler.dataOffset =
      dh.allocateArray(bufferHandler, containerHandler.m_size);
}

void cudaq::RecordLogDecoder::preallocateTuple() {
  containerHandler.dataOffset = bufferHandler.getBufferSize();
  for (auto ty : containerHandler.tupleTypes) {
    cudaq::details::DataHandlerBase &dh = getDataHandler(ty);
    containerHandler.tupleOffsets.push_back(dh.allocateTuple(bufferHandler));
  }
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
  cudaq::details::DataHandlerBase &dh = getDataHandler(label);
  dh.addRecord(bufferHandler, recValue);
}

void cudaq::RecordLogDecoder::processArrayEntry(const std::string &recValue,
                                                const std::string &recLabel) {
  std::size_t index = containerHandler.extractIndex(recLabel);
  if (index >= containerHandler.m_size)
    throw std::runtime_error("Array index out of bounds");
  cudaq::details::DataHandlerBase &dh =
      getDataHandler(containerHandler.arrayType);
  dh.insertIntoArray(bufferHandler, containerHandler.dataOffset, index,
                     recValue);
}

void cudaq::RecordLogDecoder::processTupleEntry(const std::string &recValue,
                                                const std::string &recLabel) {
  std::size_t index = containerHandler.extractIndex(recLabel);
  if (index >= containerHandler.m_size)
    throw std::runtime_error("Tuple index out of bounds");
  cudaq::details::DataHandlerBase &dh =
      getDataHandler(containerHandler.tupleTypes[index]);
  dh.insertIntoTuple(bufferHandler, containerHandler.tupleOffsets[index],
                     recValue);
}
