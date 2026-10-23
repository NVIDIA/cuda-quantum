/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file TargetAbiVersion.h
/// @brief ABI version shared by the pre-compiled target database and compiled
/// target plugin libraries.

/// Bump this whenever the `TargetConfig` struct family (TargetConfig.h), or
/// the layout `cudaq-target-db-gen` serializes it into, changes in a way
/// that is not compatible with previously-generated `TargetDatabase.gen.cpp`
/// output.
///
/// This is the link-time contract between the generator tool
/// (`cudaq-target-db-gen`, which stamps this version into the code it
/// emits) and the hand-written code in `TargetDatabase.cpp` that references
/// that exact stamped symbol name. Both sides derive the symbol name from this
/// same macro. Should they ever diverge, the linker fails outright with an
/// undefined-symbol error instead of silently misinterpreting data.
#define CUDAQ_TARGET_DB_ABI_VERSION 1

#define CUDAQ_TARGET_DB_ABI_SYMBOL_PASTE(v) cudaq_target_db_abi_v##v
#define CUDAQ_TARGET_DB_ABI_SYMBOL_EXPAND(v) CUDAQ_TARGET_DB_ABI_SYMBOL_PASTE(v)
/// The marker symbol name for the current `CUDAQ_TARGET_DB_ABI_VERSION`.
#define CUDAQ_TARGET_DB_ABI_SYMBOL_NAME                                        \
  CUDAQ_TARGET_DB_ABI_SYMBOL_EXPAND(CUDAQ_TARGET_DB_ABI_VERSION)
