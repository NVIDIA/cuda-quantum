/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/nanobind.h>

namespace py = nanobind;

namespace cudaq {
/// @brief Bind the custom operation registry to Python.
void bindCustomOpRegistry(py::module_ &mod);
} // namespace cudaq
