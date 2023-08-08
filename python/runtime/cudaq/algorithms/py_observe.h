/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>

#include "common/ObserveResult.h"
#include "cudaq/algorithms/observe.h"
#include "cudaq/builder/kernel_builder.h"
#include "cudaq/spin_op.h"

namespace py = pybind11;

namespace cudaq {

/// @brief Default shots value set to -1
inline constexpr int defaultShotsValue = -1;

/// @brief Functions for running `cudaq::observe()` from python.
/// Exposing pyObserve in the header for use elsewhere in the bindings.
observe_result pyObserve(kernel_builder<> &kernel, spin_op &spin_operator,
                         py::args args = {}, int shots = defaultShotsValue,
                         std::optional<noise_model> noise = std::nullopt);

/// @brief Expose binding of `cudaq::observe()` and `cudaq::observe_async` to
/// python.
void bindObserve(py::module &mod);
} // namespace cudaq
