/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quake, quake, cudaq::quake::QuakeDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QEC, qec, cudaq::qec::QECDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CC, cc, cudaq::cc::CCDialect)

extern "C" void cudaqRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::DialectRegistry *reg = unwrap(registry);
  cudaq::registerAllDialects(*reg);
  mlir::func::registerInlinerExtension(*reg);
  mlir::LLVM::registerInlinerInterface(*reg);
}

extern "C" void cudaqLoadAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  cudaqRegisterAllDialects(wrap(&registry));
  auto *mlirContext = unwrap(context);
  mlirContext->appendDialectRegistry(registry);
  mlirContext->loadAllAvailableDialects();
}
