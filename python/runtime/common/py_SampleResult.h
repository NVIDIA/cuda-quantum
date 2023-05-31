/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/pybind11.h>

#include "utils/LinkedLibraryHolder.h"

namespace py = pybind11;

namespace cudaq {
/// @brief Bind `cudaq.MeasureCounts` to python.
void bindMeasureCounts(py::module &mod);
} // namespace cudaq
