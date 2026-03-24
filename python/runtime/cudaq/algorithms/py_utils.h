/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>
#include <tuple>
#include <unordered_map>

namespace py = pybind11;

namespace cudaq {

/// @brief Get a JSON-encoded dictionary of a combination of all local
/// and global variables that are JSON compatible
py::dict get_serializable_var_dict();

/// @brief Fetch the Python source code from a `py::function`
std::string get_source_code(const py::function &func);

/// @brief Find the variable name for a given Python object handle. It searches
/// locally first, walks up the call stack, and finally checks the global
/// namespace. If not found, it returns an empty string.
std::string get_var_name_for_handle(const py::handle &h);

/// @brief Registry for python data classes used in kernels
class DataClassRegistry {
public:
  static std::unordered_map<std::string, std::tuple<py::object, py::dict>>
      classes;

  /// @brief Register class object
  static void registerClass(std::string &name, py::object cls) {
    classes[name] = {cls, cls.attr("__annotations__").cast<py::dict>()};
  }

  /// @brief Is data class name registered
  static bool isRegisteredClass(const std::string &name) {
    return classes.contains(name);
  }

  /// @brief Find registered data class object and its attributes
  static std::tuple<py::object, py::dict>
  getClassAttributes(std::string &name) {
    return classes[name];
  }
};

void bindPyDataClassRegistry(py::module &mod);

} // namespace cudaq
