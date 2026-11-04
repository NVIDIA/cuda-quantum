/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "runtime/mlir/py_register_dialects.h"

// Compiler-only bindings: quake/cc dialect registration and types, plus helpers
// to load intrinsics, register all dialects, and generate complex constants.
// These operate on MLIR contexts and modules without initializing execution
// runtime state. Keeping them separate from _quakeDialects lets core consumers
// construct and transform IR without loading the frontend; _quakeDialects
// re-exports these same bindings for existing frontend callers.
NB_MODULE(_quakeDialectsCore, m) { cudaq::bindRegisterDialects(m); }
