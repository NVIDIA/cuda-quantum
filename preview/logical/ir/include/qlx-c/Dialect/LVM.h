/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LVM, lvm);
#ifdef __cplusplus
}
#endif
