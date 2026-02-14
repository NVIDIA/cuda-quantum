/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RecordLogParser.h"
#include "FmtCore.h"
#include "Timing.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/runtime/logger/logger.h"

void cudaq::RecordLogParser::parse(const std::string &outputLog) {
  ScopedTraceWithContext(cudaq::TIMING_RUN, "RecordLogParser::parse");
  CUDAQ_DBG("Parsing log:\n{}", outputLog);
  std::vector<std::string> lines = cudaq::split(outputLog, '\n');
  if (lines.empty())
    return;

  // Collect log from a single shot and process it only if it is successful.
  bool processingShot = false;
  // Maintain the starting index of each shot's data
  std::size_t shotStart = 0;

  for (std::size_t idx = 0; idx < lines.size(); ++idx) {
    const auto &line = lines[idx];
    std::vector<std::string> entries = cudaq::split(line, '\t');
    if (entries.empty())
      continue;

    const std::string &recordType = entries[0];
    if (recordType == "HEADER")
      handleHeader(entries);
    else if (recordType == "METADATA")
      handleMetadata(entries);
    else if (recordType == "START") {
      processingShot = true;
      shotStart = 0;
    } else if (recordType == "OUTPUT") {
      if (processingShot)
        shotStart = shotStart == 0 ? idx : shotStart;
      else
        handleOutput(entries);
    } else if (recordType == "END") {
      if (entries.size() < 2)
        throw std::runtime_error("Missing shot status");
      if (entries[1] == "0") {
        if (processingShot) {
          // Successful shot, process it
          for (std::size_t j = shotStart; j < idx; ++j)
            handleOutput(cudaq::split(lines[j], '\t'));
        }
      } else {
        CUDAQ_DBG("Discarding shot data due to non-zero END status.");
      }
      processingShot = false;
      shotStart = 0;
      containerMeta.reset();
    } else {
      throw std::runtime_error("Invalid record type: " + recordType);
    }
  }
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
  if (entries.size() < 2 || entries.size() > 3)
    cudaq::info("Unexpected METADATA record: {}. Ignored.\n", entries);
  if (entries.size() == 3) {
    if (entries[1] == cudaq::opt::qir1_0::RequiredResultsAttrName ||
        entries[1] == cudaq::opt::qir0_1::RequiredResultsAttrName) {
      metadata[ResultCountMetadataName] = entries[2];
    } else {
      metadata[entries[1]] = entries[2];
    }
  } else {
    metadata[entries[1]] = "";
  }
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
  cudaq::trim(recLabel);
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
  if (recType == "RESULT")
    currentOutput = OutputType::RESULT;
  if (recType == "BOOL")
    currentOutput = OutputType::BOOL;
  else if (recType == "INT")
    currentOutput = OutputType::INT;
  else if (recType == "DOUBLE")
    currentOutput = OutputType::DOUBLE;
  else if (recType == "RESULT")
    currentOutput = OutputType::RESULT;
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
  static details::DataHandler<cudaq::measure_result> measureResultHandler(
      std::make_unique<details::MeasureResultConverter>());
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
  if (dataType == "result")
    return measureResultHandler;
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
  if (!dataLayoutInfo.first.has_value())
    throw std::runtime_error(
        "Data layout information missing for the struct / tuple type.");
  if (dataLayoutInfo.second.size() != containerMeta.tupleTypes.size())
    throw std::runtime_error("Tuple size mismatch in kernel and label.");
  containerMeta.dataOffset = bufferHandler.getBufferSize();
  // Directly allocate memory for the tuple, update offsets
  bufferHandler.resizeBuffer(dataLayoutInfo.first.value());
  containerMeta.tupleOffsets = dataLayoutInfo.second;
}

void cudaq::RecordLogParser::processSingleRecord(const std::string &recValue,
                                                 const std::string &recLabel) {
  auto label = recLabel;
  // For result type, we don't use the record label (register name) as the type
  // annotation.
  if (currentOutput == OutputType::RESULT)
    label = "result";
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
