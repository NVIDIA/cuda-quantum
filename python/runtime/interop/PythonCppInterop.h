/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq::python {

/// @class CppPyKernelDecorator
/// @brief A C++ wrapper for a Python object representing a CUDA-Q kernel.
class CppPyKernelDecorator {
private:
  py::object kernel;

public:
  /// @brief Constructor for CppPyKernelDecorator.
  /// @param obj A Python object representing a CUDA-Q kernel.
  /// @throw std::runtime_error if the object is not a valid CUDA-Q kernel.
  CppPyKernelDecorator(py::object obj) : kernel(obj) {
    if (!py::hasattr(obj, "compile"))
      throw std::runtime_error("Invalid python kernel object passed, must be "
                               "annotated with cudaq.kernel");
  }

  /// @brief Compiles the kernel.
  void compile() { kernel.attr("compile")(); }

  /// @brief Gets the name of the kernel.
  /// @return The name of the kernel as a string.
  std::string name() const { return kernel.attr("name").cast<std::string>(); }

  /// @brief Merges the kernel with another module.
  /// @param otherModuleStr The string representation of the other module.
  /// @return A new CppPyKernelDecorator object representing the merged kernel.
  auto merge_kernel(const std::string &otherModuleStr) {
    return CppPyKernelDecorator(kernel.attr("merge_kernel")(otherModuleStr));
  }

  /// @brief Synthesizes callable arguments for the kernel.
  /// @param name The name of the kernel.
  void synthesize_callable_arguments(const std::vector<std::string> &names) {
    kernel.attr("synthesize_callable_arguments")(names);
  }

  /// @brief Extracts a C function pointer from the kernel.
  /// @tparam `Args` Variadic template parameter for function arguments.
  /// @param kernelName The name of the kernel.
  /// @return A function pointer to the extracted C function.
  template <typename... Args>
  auto extract_c_function_pointer(const std::string &kernelName) {
    auto capsule = kernel.attr("extract_c_function_pointer")(kernelName)
                       .cast<py::capsule>();
    void *ptr = capsule;
    void (*entryPointPtr)(Args &&...) =
        reinterpret_cast<void (*)(Args &&...)>(ptr);
    return *entryPointPtr;
  }

  /// @brief Gets the Quake representation of the kernel.
  /// @return The Quake representation as a string.
  std::string get_quake() {
    return kernel.attr("__str__")().cast<std::string>();
  }
};

/// @brief Extracts the kernel name from an input MLIR string.
/// @param input The input string containing the kernel name.
/// @return The extracted kernel name.
std::string getKernelName(std::string &input);

/// @brief Extracts a sub-string from an input string based on start and end
/// delimiters.
/// @param input The input string to extract from.
/// @param startStr The starting delimiter.
/// @param endStr The ending delimiter.
/// @return The extracted sub-string.
std::string extractSubstring(const std::string &input,
                             const std::string &startStr,
                             const std::string &endStr);

/// @brief Retrieves the MLIR code and mangled kernel name for a given
/// user-level kernel name.
/// @param name The name of the kernel.
/// @return A tuple containing the MLIR code and the kernel name.
std::tuple<std::string, std::string>
getMLIRCodeAndName(const std::string &name, const std::string mangled = "");

/// @brief Register a C++ device kernel with the given module and name
/// @param module The name of the module containing the kernel
/// @param name The name of the kernel to register
void registerDeviceKernel(const std::string &module, const std::string &name,
                          const std::string &mangled);

/// @brief Retrieve the module and name of a registered device kernel
/// @param compositeName The composite name of the kernel (module.name)
/// @return A tuple containing the module name and kernel name
std::tuple<std::string, std::string>
getDeviceKernel(const std::string &compositeName);

bool isRegisteredDeviceModule(const std::string &compositeName);

template <typename T>
constexpr bool is_const_reference_v =
    std::is_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
struct TypeMangler {
  static std::string mangle() {
    std::string mangledName = typeid(T).name();
    if constexpr (is_const_reference_v<T>) {
      mangledName = "RK" + mangledName;
    }
    return mangledName;
  }
};

template <typename... Args>
std::string getMangledArgsString() {
  std::string result;
  (result += ... += TypeMangler<Args>::mangle());

  // Remove any namespace cudaq text
  std::string search = "N5cudaq";
  std::string replace = "";

  size_t pos = result.find(search);
  while (pos != std::string::npos) {
    result.replace(pos, search.length(), replace);
    pos = result.find(search, pos + replace.length());
  }

  return result;
}

/// @brief Add a C++ device kernel that is usable from CUDA-Q Python.
/// @tparam Signature The function signature of the kernel
/// @param m The Python module to add the kernel to
/// @param modName The name of the submodule to add the kernel to
/// @param kernelName The name of the kernel
/// @param docstring The documentation string for the kernel
template <typename... Signature>
void addDeviceKernelInterop(py::module_ &m, const std::string &modName,
                            const std::string &kernelName,
                            const std::string &docstring) {

  auto mangledArgs = getMangledArgsString<Signature...>();

  // FIXME Maybe Add replacement options (i.e., _pycudaq -> cudaq)

  py::module_ sub;
  if (py::hasattr(m, modName.c_str()))
    sub = m.attr(modName.c_str()).cast<py::module_>();
  else
    sub = m.def_submodule(modName.c_str());

  sub.def(
      kernelName.c_str(), [](Signature...) {}, docstring.c_str());
  cudaq::python::registerDeviceKernel(sub.attr("__name__").cast<std::string>(),
                                      kernelName, mangledArgs);
  return;
}
} // namespace cudaq::python