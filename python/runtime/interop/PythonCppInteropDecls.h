/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>

namespace cudaq::python {

/// @brief Extracts the kernel name from an input MLIR string.
/// @param input The input string containing the kernel name.
/// @return The extracted kernel name.
std::string getKernelName(const std::string &input);

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
inline std::string getMangledArgsString() {
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

template <>
inline std::string getMangledArgsString<>() {
  return {};
}

} // namespace cudaq::python
