/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "cudaq/algorithms/draw.h"
#include "utils/OpaqueArguments.h"

namespace cudaq {

/// @brief Run `cudaq::get_state` on the provided kernel and spin operator.
std::string pyDraw(kernel_builder<> &kernel, py::args args) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  OpaqueArguments argData;
  packArgs(argData, validatedArgs);

  ExecutionContext context("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&context);
  kernel.jitAndInvoke(argData.data());
  platform.reset_exec_ctx();
  return __internal__::draw(context.kernelTrace);
}

/// @brief Bind the get_state cudaq function
void bindPyDraw(py::module &mod) { mod.def("draw", &pyDraw, ""); }

} // namespace cudaq
