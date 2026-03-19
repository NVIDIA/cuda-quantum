/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "logger/pipeline/PassPipelineLogging.h"

#include "common/Environment.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

std::string jsonEscape(llvm::StringRef s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

void appendToLogFile(llvm::StringRef logPath, llvm::StringRef record) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::error_code ec;
  llvm::raw_fd_ostream logFile(
      logPath, ec, llvm::sys::fs::OF_Append | llvm::sys::fs::OF_Text);
  if (!ec)
    logFile << record << "\n";
}

} // namespace

namespace cudaq {

std::string getPipelineLogPath() {
  if (const char *path = std::getenv("CUDAQ_PIPELINE_LOG"))
    return path;
  return {};
}

/// PassInstrumentation that records which passes actually executed and writes
/// them to CUDAQ_PIPELINE_LOG as a JSONL record after the pipeline completes.
/// Attached automatically by maybeLogPassPipeline.
///
/// Each record has the form:
///   {"type":"executed","label":"<label>","passes":[{"pass":"...","op":"..."},...]}
struct PipelineRecorder : public mlir::PassInstrumentation {
  struct PassRecord {
    std::string passArg;
    std::string opName;
    bool failed = false;
  };

  std::string label;
  std::vector<PassRecord> records;

  explicit PipelineRecorder(llvm::StringRef label = {}) :
      label(label.empty() ? "<unnamed>" : label.str()) {}

  ~PipelineRecorder() override { flushToLog(); }

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    records.push_back(
        {pass->getArgument().str(), op->getName().getStringRef().str(), false});
  }

  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    if (!records.empty())
      records.back().failed = true;
  }

private:
  void flushToLog() const {
    auto logPath = getPipelineLogPath();
    if (logPath.empty())
      return;

    std::string record =
        "{\"type\":\"executed\",\"label\":\"" + jsonEscape(label) +
        "\",\"passes\":[";
    for (size_t i = 0; i < records.size(); ++i) {
      if (i > 0)
        record += ",";
      record += "{\"pass\":\"" + jsonEscape(records[i].passArg) +
                "\",\"op\":\"" + jsonEscape(records[i].opName) + "\"";
      if (records[i].failed)
        record += ",\"failed\":true";
      record += "}";
    }
    record += "]}";
    appendToLogFile(logPath, record);
  }
};

void maybeLogPassPipeline(mlir::PassManager &pm, llvm::StringRef label) {
  auto logPath = getPipelineLogPath();
  if (logPath.empty())
    return;

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  pm.printAsTextualPipeline(os);

  llvm::StringRef effectiveLabel = label.empty() ? "<unnamed>" : label;
  std::string record = "{\"type\":\"configured\",\"label\":\"" +
                       jsonEscape(effectiveLabel) + "\",\"pipeline\":\"" +
                       jsonEscape(pipeline) + "\"}";
  appendToLogFile(logPath, record);

  pm.addInstrumentation(std::make_unique<PipelineRecorder>(effectiveLabel));
}

} // namespace cudaq
