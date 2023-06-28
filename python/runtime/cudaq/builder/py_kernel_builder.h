/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>

#include "py_QuakeValue.h"
#include "utils/LinkedLibraryHolder.h"

namespace py = pybind11;

namespace cudaq {
/// @brief Function for binding `cudaq::kernel_builder` as `cudaq.Kernel`,
/// and `cudaq.make_kernel()`.
void bindBuilder(py::module &mod);
} // namespace cudaq
