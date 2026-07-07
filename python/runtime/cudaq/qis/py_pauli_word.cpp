/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_pauli_word.h"
#include "cudaq/qis/pauli_word.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace cudaq {

void bindPauliWord(nanobind::module_ &mod) {

  nanobind::class_<pauli_word>(
      mod, "pauli_word",
      "The `pauli_word` is a thin wrapper on a Pauli tensor "
      "product string, e.g. `XXYZ` on 4 qubits.")
      .def(nanobind::init<>())
      .def(nanobind::init<const std::string>())
      .def("__str__", &pauli_word::str)
      .def("__repr__",
           [](const pauli_word &w) { return "pauli_word('" + w.str() + "')"; })
      .def(
          "__eq__",
          [](const pauli_word &a, const pauli_word &b) {
            return a.str() == b.str();
          },
          nanobind::is_operator())
      .def("__hash__", [](const pauli_word &w) {
        return std::hash<std::string>{}(w.str());
      });
}
} // namespace cudaq
