/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatTimeStepper.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Internal dynamics bindings
PYBIND11_MODULE(nvqir_dynamics_bindings, m) {
  py::class_<cudaq::CuDensityMatTimeStepper>(m, "TimeStepper")
      .def(py::init<cudensitymatHandle_t, cudensitymatOperator_t>())
      .def("compute",
           [](cudaq::CuDensityMatTimeStepper &self,
              cudensitymatState_t inputState, cudensitymatState_t outputState,
              double t) {
             self.computeImpl(inputState, outputState, t, {});
           });
}
