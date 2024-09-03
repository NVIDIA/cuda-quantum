/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudaq.h"
#include "cudaq/algorithms/sample.h"
#include "quantum_lib/quantum_lib.h"
#include "runtime/interop/PythonCppInterop.h"

namespace py = pybind11;

PYBIND11_MODULE(cudaq_test_cpp_algo, m) {

  m.def("test_cpp_qalgo", [](py::object statePrepIn) {
    // Wrap the kernel and compile, will throw
    // if not a valid kernel
    cudaq::CppPyKernelDecorator statePrep(statePrepIn);
    statePrep.compile();

    // Our library exposes an "entryPoint" kernel, get its
    // mangled name and MLIR code
    auto [kernelName, cppMLIRCode] = cudaq::getMLIRCodeAndName("entryPoint");

    // Merge the entryPoint kernel into the input stateprep kernel
    auto merged = statePrep.merge_kernel(cppMLIRCode);

    // Synthesize away all callable block arguments
    merged.synthesize_callable_arguments({statePrep.name()});

    // Extract the function pointer.
    auto entryPointPtr = merged.extract_c_function_pointer(kernelName);

    // Run...
    return cudaq::sample(entryPointPtr);
  });
}