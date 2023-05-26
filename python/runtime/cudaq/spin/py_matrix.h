/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq {
/// @brief Wrapper function for exposing the bindings of `cudaq::complex_matrix`
/// to python.
void bindComplexMatrix(py::module &mod);
} // namespace cudaq
