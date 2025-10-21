/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/qis/qkernel.h"
#include "quantum_lib/quantum_lib.h"
#include "runtime/interop/PythonCppInterop.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
static std::unordered_map<std::string,
                          cudaq::qkernel<void(cudaq::qview<>, std::size_t)>>
    g_cppKernels_1;

static std::unordered_map<std::string, cudaq::qkernel<void(patch)>>
    g_cppKernels_2;

static const bool initKernels = []() {
  g_cppKernels_1.insert(std::make_pair("uccsd", cudaq::uccsd));
  g_cppKernels_2.insert(std::make_pair("reset", cudaq::reset_group));
  g_cppKernels_2.insert(std::make_pair("x", cudaq::x_group));
  return true;
}();
} // namespace

PYBIND11_MODULE(cudaq_test_cpp_algo, m) {

  m.def("test_cpp_qalgo", [](py::object statePrepIn) {
    // Wrap the kernel and compile, will throw
    // if not a valid kernel
    cudaq::python::CppPyKernelDecorator statePrep(statePrepIn);
    statePrep.compile();

    // Our library exposes an "entryPoint" kernel, get its
    // mangled name and MLIR code
    auto [kernelName, cppMLIRCode] =
        cudaq::python::getMLIRCodeAndName("entryPoint");

    // Merge the entryPoint kernel into the input stateprep kernel
    auto merged = statePrep.merge_kernel(cppMLIRCode);

    // Synthesize away all callable block arguments
    merged.synthesize_callable_arguments({statePrep.name()});

    // Extract the function pointer.
    auto entryPointPtr = merged.extract_c_function_pointer(kernelName);

    // Run...
    return cudaq::sample(entryPointPtr);
  });

  // Example of how to expose C++ kernels.
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>>(
      m, "qstd", "qft", "(Fake) Quantum Fourier Transform.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "another", "Demonstrate we can have multiple ones.");

  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "uccsd", "");

  // Convert the C++ kernel registry to Python-accessible kernels
  auto interopSubMod = m.def_submodule("_cpp_interop_kernels");
  static std::unordered_map<std::string, py::object> g_py_kernels;

  for (auto &[name, kernel] : g_cppKernels_1) {
    const char *qkernelName = cudaq::registry::getLinkableKernelNameOrNull(
        cudaq::registry::__cudaq_getLinkableKernelKey(&kernel));
    if (!qkernelName) {
      throw std::runtime_error("Could not find registered kernel name for " +
                               name);
    }

    std::string kernelName = qkernelName;
    if (kernelName.starts_with("function_"))
      kernelName = kernelName.substr(std::string("function_").length());

    interopSubMod.def(
        kernelName.c_str(), [](py::object qview, std::size_t i) {},
        "Auto-generated one-qubit encoding kernel from C++ code");
    cudaq::python::registerDeviceKernel(
        interopSubMod.attr("__name__").cast<std::string>(), kernelName, "");
    g_py_kernels.insert(
        std::make_pair(name, interopSubMod.attr(kernelName.c_str())));
  }

  for (auto &[name, kernel] : g_cppKernels_2) {
    const char *qkernelName = cudaq::registry::getLinkableKernelNameOrNull(
        cudaq::registry::__cudaq_getLinkableKernelKey(&kernel));
    if (!qkernelName) {
      throw std::runtime_error("Could not find registered kernel name for " +
                               name);
    }

    std::string kernelName = qkernelName;
    if (kernelName.starts_with("function_"))
      kernelName = kernelName.substr(std::string("function_").length());

    interopSubMod.def(
        kernelName.c_str(), [](py::object patch) {},
        "Auto-generated one-qubit encoding kernel from C++ code");
    cudaq::python::registerDeviceKernel(
        interopSubMod.attr("__name__").cast<std::string>(), kernelName, "");
    g_py_kernels.insert(
        std::make_pair(name, interopSubMod.attr(kernelName.c_str())));
  }

  m.def("get_cpp_kernel", [](const std::string &name) {
    auto it = g_py_kernels.find(name);
    if (it == g_py_kernels.end())
      throw std::runtime_error("No C++ kernel registered for requested name.");

    return it->second;
  });
}
