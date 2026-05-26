/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "PythonCppInteropDecls.h"
#include "cudaq/qis/qkernel.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace cudaq::python {

/// @brief A C++ wrapper for a Python object representing a CUDA-Q kernel.
class CppPyKernelDecorator {
public:
  /// The constructor.
  /// @param obj A kernel decorator Python object.
  /// @throw std::runtime_error if the object is not a valid kernel decorator.
  CppPyKernelDecorator(nanobind::object obj) : kernel(obj) {
    if (!nanobind::hasattr(obj, "qkeModule"))
      throw std::runtime_error("Invalid python kernel object passed, must be "
                               "annotated with cudaq.kernel");
  }

  ~CppPyKernelDecorator() = default;

  /// Fully compiles this python kernel, returning a `qkernel` that can
  /// be directly invoked by host code. Do not pass the returned `qkernel`
  /// into other kernels, as this will lead to bad things.
  template <typename T, typename... As>
    requires QKernelType<T>
  T getEntryPointFunction(As... as) {
    auto p = getKernelHelper(/*isEntryPoint=*/true, as...);
    auto *fptr = reinterpret_cast<typename T::function_type *>(p);
    return T{fptr};
  }

  /// Fully compiles this python kernel, returning a `qkernel` that can
  /// be indirectly invoked by other kernels. Only pass the returned
  /// `qkernel` into other kernels, calling it directly from host code
  /// will lead to bad things.
  template <typename T, typename... As>
    requires QKernelType<T>
  T getDirectKernelCall(As... as) {
    auto p = getKernelHelper(/*isEntryPoint=*/false, as...);
    auto *fptr = reinterpret_cast<typename T::function_type *>(p);
    return T{fptr};
  }

private:
  nanobind::object kernel;
  // Hold on to the CompiledModule, it keeps the JIT engine alive.
  nanobind::object compiledKernel;

  template <typename... As>
  void *getKernelHelper(bool isEntryPoint, As... as) {
    // Perform beta reduction on the kernel decorator.
    compiledKernel =
        kernel.attr("beta_reduction")(isEntryPoint, std::forward<As>(as)...);
    auto entryPointAddr =
        nanobind::cast<std::uintptr_t>(compiledKernel.attr("entry_point"));
    // Set lsb to 1 to denote this is NOT a C++ kernel.
    auto *p = reinterpret_cast<void *>(
        static_cast<std::intptr_t>(entryPointAddr) | 1);
    // Translate the pointer to the entry point code buffer to a `qkernel`.
    return p;
  }
};

/// This template allows a single python decorator to be called from a C++
/// function (i.e., the algorithm). The actual arguments are specialized
/// (synthesized) into the kernel and cannot be changed by the algorithm.
template <typename KT, typename ALGO, typename... As>
  requires QKernelType<KT> && std::invocable<ALGO, KT>
auto launch_specialized_py_decorator(nanobind::object qern, ALGO algo,
                                     As... as) {
  cudaq::python::CppPyKernelDecorator decorator(qern);
  auto entryPoint = decorator.getDirectKernelCall<KT>(std::forward<As>(as)...);
  return algo(std::move(entryPoint));
}

/// @brief Add a C++ device kernel that is usable from CUDA-Q Python.
/// @tparam Signature The function signature of the kernel
/// @param m The Python module to add the kernel to
/// @param modName The name of the submodule to add the kernel to
/// @param kernelName The name of the kernel
/// @param docstring The documentation string for the kernel
template <typename... Signature>
void addDeviceKernelInterop(nanobind::module_ &m, const std::string &modName,
                            const std::string &kernelName,
                            const std::string &docstring) {

  auto mangledArgs = getMangledArgsString<Signature...>();

  // FIXME Maybe Add replacement options (i.e., _pycudaq -> cudaq)

  nanobind::module_ sub =
      nanobind::hasattr(m, modName.c_str())
          ? nanobind::cast<nanobind::module_>(m.attr(modName.c_str()))
          : m.def_submodule(modName.c_str());

  sub.def(kernelName.c_str(), [](Signature...) {}, docstring.c_str());
  cudaq::python::registerDeviceKernel(
      nanobind::cast<std::string>(sub.attr("__name__")), kernelName,
      mangledArgs);
  return;
}
} // namespace cudaq::python
