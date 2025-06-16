/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "py_ResourceCounts.h"

#include "common/ResourceCounts.h"

#include <sstream>

namespace cudaq {

void bindResourceCounts(py::module &mod) {
  using namespace cudaq;

  // TODO Bind the variants of this functions that take the register name
  // as input.
  py::class_<resource_counts>(
      mod, "ResourceCounts",
      R"#(A data-type containing the results of a call to :func:`resource_count`. 
This includes all gate counts.)#")
      .def(py::init<>())
      .def(
          "dump", [](resource_counts &self) { self.dump(); },
          "Print a string of the raw resource counts data to the "
          "terminal.\n")
      .def(
          "count",
          [](resource_counts &self, const std::string gate) {
            return self.count(gate);
          },
          "Get the number of occurrences of a given gate")
      .def(
          "__str__",
          [](resource_counts &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Return a string of the raw resource counts that are stored in "
          "`self`.\n")
      .def("clear", &resource_counts::clear,
           "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
