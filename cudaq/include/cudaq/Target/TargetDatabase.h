/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetAbiVersion.h"
#include "cudaq/Target/TargetConfig.h"
#include <string_view>
#include <utility>
#include <vector>

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
