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
It carries the measurement value and an optional unique identifier
for backend-specific metadata correlation.)#")
      .def(py::init<int64_t>(), py::arg("value"))
      .def(py::init<int64_t, int64_t>(), py::arg("value"), py::arg("unique_id"))
      .def_readonly("value", &measure_result::value,
                    "The integer measurement value (0 or 1).")
      .def_readonly("unique_id", &measure_result::unique_id,
                    "The unique identifier for this measurement "
                    "result (INT64_MAX if unassigned).")
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
        os << "measure_result(value=" << self.value << ", id=" << self.unique_id
           << ")";
        return os.str();
      });
}
} // namespace cudaq
