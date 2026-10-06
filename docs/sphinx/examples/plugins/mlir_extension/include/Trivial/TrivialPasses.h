/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#pragma once

#include "mlir/Pass/Pass.h"

namespace trivial {

// TableGen-generated pass declarations + registration hooks.
// `createTrivialPass` is emitted here by `-gen-pass-decls` (see
// include/Trivial/CMakeLists.txt).
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Trivial/TrivialPasses.h.inc"

} // namespace trivial
