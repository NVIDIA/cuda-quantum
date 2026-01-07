/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_qubit_qis.h"
#include "cudaq/qis/qubit_qis.h"
#include <fmt/core.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace cudaq {

void bindQIS(py::module &mod) {

  py::class_<qubit>(
      mod, "qubit",
      "The qubit is the primary unit of information in a quantum computer. "
      "Qubits can be created individually or as part of larger registers.")
      .def(py::init<>())
      .def(
          "__invert__", [](qubit &self) -> qubit & { return !self; },
          "Negate the control qubit.")
      .def("is_negated", &qubit::is_negative,
           "Returns true if this is a negated control qubit.")
      .def("reset_negation", &qubit::negate,
           "Removes the negated state of a control qubit.")
      .def(
          "id", [](qubit &self) { return self.id(); },
          "Return a unique integer identifier for this qubit.");

  py::class_<qview<>>(mod, "qview",
                      "A non-owning view on a register of qubits.")
      .def(
          "size", [](qview<> &self) { return self.size(); },
          "Return the number of qubits in this view.")
      .def(
          "front", [](qview<> &self) -> qubit & { return self.front(); },
          "Return first qubit in this view.")
      .def(
          "front",
          [](qview<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this view.")
      .def(
          "back", [](qview<> &self) -> qubit & { return self.back(); },
          "Return the last qubit in this view.")
      .def(
          "back",
          [](qview<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this view.")
      .def(
          "__iter__",
          [](qview<> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "slice",
          [](qview<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qview<>::operator[],
           py::return_value_policy::reference,
           "Return the qubit at the given index.");

  py::class_<qvector<>>(
      mod, "qvector",
      "An owning, dynamically sized container for qubits. The semantics of the "
      "`qvector` follows that of a `std::vector` or `list` for qubits.")
      .def(py::init<std::size_t>())
      .def(
          "size", [](qvector<> &self) { return self.size(); },
          "Return the number of qubits in this `qvector`.")
      .def(
          "front",
          [](qvector<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this `qvector` as a non-owning view.")
      .def(
          "front", [](qvector<> &self) -> qubit & { return self.front(); },
          py::return_value_policy::reference,
          "Return first qubit in this `qvector`.")
      .def(
          "back", [](qvector<> &self) -> qubit & { return self.back(); },
          py::return_value_policy::reference,
          "Return the last qubit in this `qvector`.")
      .def(
          "back",
          [](qvector<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this `qvector` as a non-owning "
          "view.")
      .def(
          "__iter__",
          [](qvector<> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "slice",
          [](qvector<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qvector<2>::operator[],
           py::return_value_policy::reference,
           "Return the qubit at the given index.");

  py::class_<pauli_word>(mod, "pauli_word",
                         "The `pauli_word` is a thin wrapper on a Pauli tensor "
                         "product string, e.g. `XXYZ` on 4 qubits.")
      .def(py::init<>())
      .def(py::init<const std::string>());
}
} // namespace cudaq
