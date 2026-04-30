/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <nanobind/nanobind.h>
#include <string>
#include <tuple>
#include <unordered_map>

namespace cudaq {

/// @brief Get a JSON-encoded dictionary of a combination of all local
/// and global variables that are JSON compatible
nanobind::dict get_serializable_var_dict();

/// @brief Fetch the Python source code from a `nanobind::callable`
std::string get_source_code(const nanobind::callable &func);

/// @brief Find the variable name for a given Python object handle. It searches
/// locally first, walks up the call stack, and finally checks the global
/// namespace. If not found, it returns an empty string.
std::string get_var_name_for_handle(const nanobind::handle &h);

/// @brief Registry for python data classes used in kernels
class DataClassRegistry {
public:
  static std::unordered_map<std::string,
                            std::tuple<nanobind::object, nanobind::dict>>
      classes;

  /// @brief Register class object
  static void registerClass(std::string &name, nanobind::object cls) {
    classes[name] = {
        cls, nanobind::cast<nanobind::dict>(cls.attr("__annotations__"))};
  }

  /// @brief Is data class name registered
  static bool isRegisteredClass(const std::string &name) {
    return classes.contains(name);
  }

  /// @brief Find registered data class object and its attributes
  static std::tuple<nanobind::object, nanobind::dict>
  getClassAttributes(std::string &name) {
    return classes[name];
  }
};

void bindPyDataClassRegistry(nanobind::module_ &mod);

} // namespace cudaq
