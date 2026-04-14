/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/measure_result.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

namespace cudaq {

void bindMeasureResult(py::module &mod) {
  py::class_<measure_result>(mod, "measure_result",
                             R"#(A quantum measurement result.
This type represents the outcome of a quantum measurement operation.
It is not user-constructible; instances originate from mz/mx/my calls
within quantum kernels.)#")
      .def_readonly("value", &measure_result::value,
                    "The integer measurement value (0 or 1).")
      .def("__bool__",
           [](const measure_result &self) { return static_cast<bool>(self); })
      .def("__int__", [](const measure_result &self) { return self.value; })
      .def("__eq__", [](const measure_result &self,
                        const measure_result &other) { return self == other; })
      .def("__eq__",
           [](const measure_result &self, bool other) {
             return static_cast<bool>(self) == other;
           })
      .def("__ne__", [](const measure_result &self,
                        const measure_result &other) { return self != other; })
      .def("__ne__",
           [](const measure_result &self, bool other) {
             return static_cast<bool>(self) != other;
           })
      .def("__repr__", [](const measure_result &self) {
        std::ostringstream os;
        os << "measure_result(value=" << self.value << ")";
        return os.str();
      });
}
} // namespace cudaq
