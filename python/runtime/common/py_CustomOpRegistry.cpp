/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "py_CustomOpRegistry.h"
#include "common/CustomOp.h"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace cudaq {
struct py_unitary_operation : public unitary_operation {
  std::vector<std::complex<double>>
  unitary(const std::vector<double> &parameters =
              std::vector<double>()) const override {
    throw std::runtime_error("Attempt to invoke the placeholder for Python "
                             "unitary op. This is illegal.");
    return {};
  }
};

void bindCustomOpRegistry(py::module &mod) {
  mod.def(
      "register_custom_operation",
      [&](const std::string &opName) {
        cudaq::customOpRegistry::getInstance()
            .registerOperation<py_unitary_operation>(opName);
      },
      "Register a custom operation");
}
} // namespace cudaq
