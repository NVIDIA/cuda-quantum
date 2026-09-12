/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Packages the CUDA-Q Quake and CC dialects as a loadable MLIR DialectPlugin,
// loaded at runtime by the qlx Python extension (_qlxRuntime.load_plugin).
// The plugin links cudaq::MLIR, so all MLIR and Quake/CC symbols resolve from
// libcudaqMLIR through its own DT_NEEDED entries; no RTLD_GLOBAL promotion of
// host images is required.
//
// The Quake/CC definitions and TypeIDs resolve from CUDA-Q's shared MLIR
// target, so CUDA-Q and QLX operate in one compiler universe. Static CUDA-Q
// dialect archives are intentionally unsupported by this plugin.
//
// This is a _qlxRuntime plugin only -- not a qlx-opt --load-*-plugin target
// (qlx-opt already links the pass and its configured Quake/CC provider).
//
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

extern "C" LLVM_ATTRIBUTE_WEAK
    __attribute__((visibility("default"))) ::mlir::DialectPluginLibraryInfo
    mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "QuakeImport", "v1",
          [](mlir::DialectRegistry *reg) {
            reg->insert<cudaq::cc::CCDialect, cudaq::quake::QuakeDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK
    __attribute__((visibility("default"))) ::mlir::PassPluginLibraryInfo
    mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "QuakeImport", "v1", []() {}};
}
