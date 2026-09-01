//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//

#include "qlx-c/Dialect/LVM.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "mlir/CAPI/Registration.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LVM, lvm, qlx::lvm::LVMDialect)
