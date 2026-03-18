/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Environment.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include <string>

namespace cudaq {

inline bool shouldPrintMLIRPassPipeline() {
#ifdef CUDAQ_DEBUG
  // Keep this debug-only so release builds avoid pipeline serialization work
  // even if the environment variable is set.
  return getEnvBool("CUDAQ_MLIR_PRINT_PASS_PIPELINE", false);
#else
  return false;
#endif
}

inline void maybeLogPassPipeline(mlir::PassManager &pm,
                                 llvm::StringRef label = {}) {
  if (!shouldPrintMLIRPassPipeline())
    return;

  // TODO: pass statistics is canonical way to do this sort of testing
  // pm.enableStatistics()

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  pm.printAsTextualPipeline(os);

  if (pipeline.empty())
    pipeline = "<empty>";

  if (label.empty())
    llvm::outs() << "MLIR pass pipeline:\n" << pipeline << "\n";
  else
    llvm::outs() << "MLIR pass pipeline (" << label << "):\n" << pipeline << "\n";
}

} // namespace cudaq
