/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nvqir/AnalysisScope.h"
#include <string>

namespace nvqir::dem {

/// @brief Activate the DEM (Detector Error Model) analysis on the current
/// thread, backed by a simulator plugin.
///
/// The returned `AnalysisScope` claims the thread-local analysis slot,
/// resets the recorded circuit of the backend on entry, and releases the slot
/// on destruction. While the scope is alive, every gate, measurement, noise
/// channel, and `qec.*` ops lowered out of the kernel is appended
/// to the simulator's recorded circuit. `cudaq::dem_from_kernel` consumes
/// the recorded circuit after the kernel returns and feeds it to Stim's
/// `ErrorAnalyzer`.
///
/// @param plugin_name  NVQIR simulator plugin to drive the analysis. Defaults
///                     to `stim`
///
/// Throws `std::runtime_error` if an analysis scope is already active on the
/// current thread, if the plugin shared library cannot be loaded, or if the
/// plugin's simulator does not implement `nvqir::RecordedCircuit`.
AnalysisScope make_scope(std::string plugin_name = "stim");

} // namespace nvqir::dem
