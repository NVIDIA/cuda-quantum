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
      .def(py::init([](int64_t handlePtrInt, int64_t operatorPtrInt) {
        cudensitymatHandle_t handle =
            reinterpret_cast<cudensitymatHandle_t>(handlePtrInt);
        cudensitymatOperator_t op =
            reinterpret_cast<cudensitymatOperator_t>(operatorPtrInt);
        return cudaq::CuDensityMatTimeStepper(handle, op);
      }))
      .def("compute",
           [](cudaq::CuDensityMatTimeStepper &self, int64_t inputStatePtr,
              int64_t outputStatePtr, double t) {
             cudensitymatState_t inputState =
                 reinterpret_cast<cudensitymatState_t>(inputStatePtr);
             cudensitymatState_t outputState =
                 reinterpret_cast<cudensitymatState_t>(outputStatePtr);
             self.computeImpl(inputState, outputState, t, {});
           });
}
