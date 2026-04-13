/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <nanobind/nanobind.h>
#include <tuple>
#include <unordered_map>

namespace nb = nanobind;

namespace cudaq {

/// @brief Get a JSON-encoded dictionary of a combination of all local
/// and global variables that are JSON compatible
nb::dict get_serializable_var_dict();

/// @brief Fetch the Python source code from a `nb::callable`
std::string get_source_code(const nb::callable &func);

/// @brief Find the variable name for a given Python object handle. It searches
/// locally first, walks up the call stack, and finally checks the global
/// namespace. If not found, it returns an empty string.
std::string get_var_name_for_handle(const nb::handle &h);

/// @brief Registry for python data classes used in kernels
class DataClassRegistry {
public:
  static std::unordered_map<std::string, std::tuple<nb::object, nb::dict>>
      classes;

  /// @brief Register class object
  static void registerClass(std::string &name, nb::object cls) {
    classes[name] = {cls, nb::cast<nb::dict>(cls.attr("__annotations__"))};
  }

  /// @brief Is data class name registered
  static bool isRegisteredClass(const std::string &name) {
    return classes.contains(name);
  }

  /// @brief Find registered data class object and its attributes
  static std::tuple<nb::object, nb::dict>
  getClassAttributes(std::string &name) {
    return classes[name];
  }
};

void bindPyDataClassRegistry(nb::module_ &mod);

} // namespace cudaq
