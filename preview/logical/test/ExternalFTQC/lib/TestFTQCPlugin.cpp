/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "test_ftqc/TestFTQCDialect.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"

extern "C" LLVM_ATTRIBUTE_WEAK
    __attribute__((visibility("default"))) ::mlir::DialectPluginLibraryInfo
    mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TestFTQC", "v1",
          [](mlir::DialectRegistry *registry) {
            registry->insert<test_ftqc::TestFTQCDialect>();
          }};
}
