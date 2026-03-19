/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include <string>
#include <vector>

namespace cudaq {

/// Returns the pipeline log file path from the CUDAQ_PIPELINE_LOG environment
/// variable, or an empty string if not set.
std::string getPipelineLogPath();

/// Log the configured pass pipeline to CUDAQ_PIPELINE_LOG as a JSONL record
/// and attach a PipelineRecorder instrumentation to capture what actually runs.
/// Call this after adding passes to the PassManager but before pm.run().
///
/// Emits two JSONL records per pipeline:
///   {"type":"configured","label":"<label>","pipeline":"<textual-pipeline>"}
///   {"type":"executed","label":"<label>","passes":[{"pass":"...","op":"..."},...]}
void maybeLogPassPipeline(mlir::PassManager &pm, llvm::StringRef label = {});

} // namespace cudaq
