/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "py_Resources.h"

#include "common/Resources.h"

#include <sstream>

namespace cudaq {

void bindResources(nanobind::module_ &mod) {
  using namespace cudaq;

  nanobind::class_<Resources>(
      mod, "Resources",
      R"#(A data-type containing the results of a call to :func:`estimate_resources`.
This includes all gate counts.)#")
      .def(nanobind::init<>())
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
      .def_prop_ro("num_qubits", &Resources::getNumQubits,
                   "The total number of qubits allocated in the kernel.\n")
      .def_prop_ro("num_used_qubits", &Resources::getNumUsedQubits,
                   "The number of qubits touched by at least one quantum "
                   "operation.\n")
      .def_prop_ro("depth", &Resources::getCircuitDepth,
                   "The circuit depth (longest gate chain on any qubit).\n")
      .def_prop_ro(
          "gate_count_by_arity",
          [](Resources &self) { return self.getGateCountsByArity(); },
          "Gate counts by qubit arity, as a dict mapping arity to count.\n")
      .def("gate_count_for_arity", &Resources::getGateCountByArity,
           nanobind::arg("arity"),
           "Get gate count for a specific qubit arity (total qubits "
           "including controls and targets). Returns 0 if no gates of "
           "that arity exist.")
      .def("depth_for_arity", &Resources::getDepthByArity,
           nanobind::arg("arity"),
           "Get circuit depth considering only gates of a specific qubit "
           "arity. Returns 0 if no gates of that arity exist.")
      .def_prop_ro("multi_qubit_gate_count", &Resources::getMultiQubitGateCount,
                   "Total count of gates with 2 or more qubits.\n")
      .def_prop_ro("multi_qubit_depth", &Resources::getMultiQubitDepth,
                   "Max depth across all gate widths >= 2.\n")
      .def_prop_ro(
          "per_qubit_depth",
          [](Resources &self) { return self.getPerQubitDepth(); },
          "Per-qubit circuit depth (all gates), as a dict mapping qubit "
          "index to depth.\n")
      .def("clear", &Resources::clear, "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
