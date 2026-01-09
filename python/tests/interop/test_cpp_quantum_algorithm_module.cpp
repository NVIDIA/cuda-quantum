/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/algorithms/sample.h"
#include "quantum_lib/quantum_lib.h"
#include "runtime/interop/PythonCppInterop.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cudaq_test_cpp_algo, m) {
  // Example of how to expose C++ kernels.
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>>(
      m, "qstd", "qft", "(Fake) Quantum Fourier Transform.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "another", "Demonstrate we can have multiple ones.");

  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "uccsd", "");
}
