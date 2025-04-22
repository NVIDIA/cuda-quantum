/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace cudaq {

inline std::size_t byteSize(mlir::Type ty) {
  if (isa<mlir::ComplexType>(ty)) {
    auto eleTy = cast<mlir::ComplexType>(ty).getElementType();
    return 2 * cudaq::opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
  }
  if (ty.isIntOrFloat())
    return cudaq::opt::convertBitsToBytes(ty.getIntOrFloatBitWidth());
  ty.dump();
  throw std::runtime_error("Expected a complex, floating, or integral type");
}

/// @brief Convert raw data to python object.
py::object convertResult(mlir::Type ty, char *data, std::size_t size);

/// @brief Launch python kernel with arguments.
void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs,
                       const std::vector<std::string> &names);

void bindAltLaunchKernel(py::module &mod);
} // namespace cudaq
