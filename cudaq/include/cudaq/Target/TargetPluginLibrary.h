/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"

namespace cudaq::config {

/// The exported C symbol name a compiled external target plugin library must
/// define. Consumers will `dlopen()` the library and `dlsym()` exactly this
/// name.
///
/// This name is the load-time equivalent of a link-time ABI version check: a
/// plugin library built against a different `TargetConfig`/generator ABI
/// exports (or is looked up under) a different symbol name entirely, so a
/// mismatch fails immediately and unambiguously at `dlsym` time.
inline constexpr const char *kTargetPluginSymbolName = "cudaq_target_config_v1";

/// Signature of the symbol named by `kTargetPluginSymbolName`.
using TargetPluginEntryPoint = const TargetConfig *(*)();

} // namespace cudaq::config
