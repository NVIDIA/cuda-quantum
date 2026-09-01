//===- qlx-translate.cpp - QLX translation tool ------------------*- C++
//-*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// qlx-translate: translates between QLX MLIR and external formats.
//
// Currently supported:
//   --fabric-to-stim   Translate Fabric MLIR to Stim circuit text
//
// Usage:
//   qlx-translate --fabric-to-stim input.mlir
//   qlx-translate --fabric-to-stim input.mlir -o output.stim
//
//===----------------------------------------------------------------------===//

#include "qlx/Target/Fabric/EmitStim.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <cstdlib>
#include <memory>

int main(int argc, char **argv) {
  qlx::fabric::registerFabricToStimTranslation();
  // Translate the LogicalResult to an int exit code matching the
  // convention used by upstream MLIR tools (EXIT_SUCCESS=0,
  // EXIT_FAILURE=1). This preserves more nuanced status from
  // `mlirTranslateMain` than the bool conversion of `mlir::failed`
  // would.
  return mlir::failed(
             mlir::mlirTranslateMain(argc, argv, "QLX translation tool"))
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}
