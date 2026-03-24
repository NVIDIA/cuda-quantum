/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_pauli_word.h"
#include "cudaq/qis/pauli_word.h"
#include <pybind11/pybind11.h>

namespace cudaq {

void bindPauliWord(py::module &mod) {

  py::class_<pauli_word>(mod, "pauli_word",
                         "The `pauli_word` is a thin wrapper on a Pauli tensor "
                         "product string, e.g. `XXYZ` on 4 qubits.")
      .def(py::init<>())
      .def(py::init<const std::string>());
}
} // namespace cudaq
