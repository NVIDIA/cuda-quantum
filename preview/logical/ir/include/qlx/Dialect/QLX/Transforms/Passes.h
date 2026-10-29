/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "mlir/Pass/Pass.h"

namespace qlx {

#define GEN_PASS_DECL
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"

} // namespace qlx
