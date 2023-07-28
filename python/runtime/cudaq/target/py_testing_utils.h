/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq {

class LinkedLibraryHolder;

/// @brief Bind test utilities needed for mock qpu base profile simulation
void bindTestUtils(py::module &mod, LinkedLibraryHolder &holder);

} // namespace cudaq
