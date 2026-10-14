/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include <string_view>
#include <utility>
#include <vector>

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

extern "C" void CUDAQ_TARGET_DB_ABI_SYMBOL_NAME();

namespace cudaq::config {

inline constexpr unsigned kTargetDatabaseAbiVersion =
    CUDAQ_TARGET_DB_ABI_VERSION;

/// Look up an in-tree target by name in the pre-compiled target database that
/// is generated at build time. Returns `nullptr` if `name` is not one of those
/// built-in targets.
///
/// The returned pointer is valid for the lifetime of the program.
const TargetConfig *lookupBuiltinTarget(std::string_view name);

/// Every in-tree target known to the pre-compiled target database, as
/// `{name, config}` pairs.
const std::vector<std::pair<std::string_view, const TargetConfig *>> &
listBuiltinTargets();

namespace detail {
/// Accessor for the generated `{name, config}` table, defined in the
/// build-time-generated translation unit produced by `cudaq-target-db-gen`.
/// Not for direct use outside of `TargetDatabase.cpp`.
const std::vector<std::pair<std::string_view, const TargetConfig *>> &
builtinTargetTable();
} // namespace detail

} // namespace cudaq::config
