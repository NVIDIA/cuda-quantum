/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// CUDA-Q MLIR Python site initializer.

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "CUDA-Q MLIR site initializer (default dialect registration).";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    cudaqRegisterAllDialects(registry);
  });
}
