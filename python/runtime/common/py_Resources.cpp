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
      .def_property_readonly(
          "num_qubits", &Resources::getNumQubits,
          "The total number of qubits allocated in the kernel.\n")
      .def_property_readonly(
          "num_used_qubits", &Resources::getNumUsedQubits,
          "The number of qubits touched by at least one quantum "
          "operation.\n")
      .def_property_readonly(
          "depth", &Resources::getCircuitDepth,
          "The circuit depth (longest gate chain on any qubit).\n")
      .def_property_readonly(
          "gate_count_by_arity",
          [](Resources &self) {
            return py::dict(py::cast(self.getGateCountsByArity()));
          },
          "Gate counts by qubit arity, as a dict mapping arity to count.\n")
      .def("gate_count_for_arity", &Resources::getGateCountByArity,
           py::arg("arity"),
           "Get gate count for a specific qubit arity (total qubits "
           "including controls and targets). Returns 0 if no gates of "
           "that arity exist.")
      .def("depth_for_arity", &Resources::getDepthByArity, py::arg("arity"),
           "Get circuit depth considering only gates of a specific qubit "
           "arity. Returns 0 if no gates of that arity exist.")
      .def_property_readonly("multi_qubit_gate_count",
                             &Resources::getMultiQubitGateCount,
                             "Total count of gates with 2 or more qubits.\n")
      .def_property_readonly("multi_qubit_depth",
                             &Resources::getMultiQubitDepth,
                             "Max depth across all multi-qubit arities.\n")
      .def_property_readonly(
          "per_qubit_depth",
          [](Resources &self) {
            return py::dict(py::cast(self.getPerQubitDepth()));
          },
          "Per-qubit circuit depth (all gates), as a dict mapping qubit "
          "index to depth.\n")
      .def("clear", &Resources::clear, "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
