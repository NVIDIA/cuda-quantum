/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq {
void bindPyToStim(py::module &mod, LinkedLibraryHolder &holder);
} // namespace cudaq
