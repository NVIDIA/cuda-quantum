/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>
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
  static std::unordered_map<std::string, py::object> classes;

  /// @brief Register class object
  static void registerClass(std::string &name, py::object cls) {
    classes[name] = cls;
  }

  /// @brief Is data class name registered
  static bool isRegisteredClass(const std::string &name) {
    return classes.contains(name);
  }

  /// @brief Find registered data class object
  static py::object getClass(std::string &name) { return classes[name]; }

  /// @brief Find registered data class object and its attributes
  static py::tuple getClassAttributes(std::string &name) {
    py::list list;
    py::object cls = getClass(name);
    list.append(getClass(name));
    list.append(getAttributes(cls));
    return py::tuple(list);
  }

  /// @brief Find class attributes
  static py::dict getAttributes(py::object cls) {
    return cls.attr("__annotations__").cast<py::dict>();
  }
};

void bindPyDataClassRegistry(py::module &mod);

} // namespace cudaq
