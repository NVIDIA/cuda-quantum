/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/IR/QTXDialect.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace cudaq {
void registerToOpenQASMTranslation();
void registerToIQMJsonTranslation();
} // namespace cudaq

int main(int argc, char **argv) {
  cudaq::registerToOpenQASMTranslation();
  cudaq::registerToIQMJsonTranslation();

  mlir::DialectRegistry registry;
  registry.insert<qtx::QTXDialect>();
  return failed(
      mlir::mlirTranslateMain(argc, argv, "QTX Translation Testing Tool"));
}
