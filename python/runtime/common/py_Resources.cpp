/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "py_Resources.h"

#include "common/Resources.h"

#include <sstream>

namespace cudaq {

void bindResources(py::module &mod) {
  using namespace cudaq;

  py::class_<Resources>(
      mod, "Resources",
      R"#(A data-type containing the results of a call to :func:`estimate_resources`. 
This includes all gate counts.)#")
      .def(py::init<>())
      .def(
          "dump", [](Resources &self) { self.dump(); },
          "Print a string of the raw resource counts data to the "
          "terminal.\n")
      .def(
          "count_controls",
          [](Resources &self, const std::string &gate, size_t nControls) {
            return self.count_controls(gate, nControls);
          },
          "Get the number of occurrences of a given gate with the given number "
          "of controls")
      .def(
          "count",
          [](Resources &self, const std::string &gate) {
            return self.count(gate);
          },
          "Get the number of occurrences of a given gate with any number of "
          "controls")
      .def(
          "count", [](Resources &self) { return self.count(); },
          "Get the total number of occurrences of all gates")
      .def(
          "__str__",
          [](Resources &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Return a string of the raw resource counts that are stored in "
          "`self`.\n")
      .def(
          "to_dict", [](Resources &self) { return self.gateCounts(); },
          "Return a dictionary of the raw resource counts that are stored in "
          "`self`.\n")
      .def("clear", &Resources::clear, "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
