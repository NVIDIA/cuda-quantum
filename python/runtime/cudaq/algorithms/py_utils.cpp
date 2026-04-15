/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_utils.h"
#include "cudaq/utils/cudaq_utils.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace cudaq {

py::dict get_serializable_var_dict() {
  py::object json = py::module_::import_("json");
  py::dict serialized_dict;

  auto try_to_add_item = [&](const auto item) {
    try {
      auto key = item.first;
      auto value = item.second;

      std::string keyStr(py::str(key).c_str());
      if (keyStr.starts_with("__")) {
        // Ignore items that start with "__" (like Python __builtins__, etc.)
      } else if (py::hasattr(value, "to_json")) {
        auto type =
            py::handle(reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())));
        std::string module(py::str(type.attr("__module__")).c_str());
        std::string name(py::str(type.attr("__name__")).c_str());
        auto type_name = py::str((module + "." + name).c_str());
        py::str json_key_name((keyStr + "/" + module + "." + name).c_str());
        serialized_dict[json_key_name] =
            json.attr("loads")(value.attr("to_json")());
      } else if (py::hasattr(value, "tolist")) {
        serialized_dict[key] =
            json.attr("loads")(json.attr("dumps")(value.attr("tolist")()));
      } else {
        serialized_dict[key] = json.attr("loads")(json.attr("dumps")(value));
      }
    } catch (const py::python_error &e) {
      // Serialization failures are non-fatal - we just skip the entry.
    }
  };

  for (const auto item : py::globals())
    try_to_add_item(item);

  py::object inspect = py::module_::import_("inspect");
  std::vector<py::object> frame_vec;
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    frame_vec.push_back(py::object(current_frame));
    current_frame = current_frame.attr("f_back");
  }

  for (auto it = frame_vec.rbegin(); it != frame_vec.rend(); ++it) {
    py::dict f_locals = it->attr("f_locals");
    for (const auto item : f_locals)
      try_to_add_item(item);
  }

  return serialized_dict;
}

// Find the minimum indent level for a set of lines in string and remove them
// from every line in the string.
static std::size_t strip_leading_whitespace(std::string &source_code) {
  std::size_t min_indent = std::numeric_limits<std::size_t>::max();

  auto lines = cudaq::split(source_code, '\n');
  for (auto &line : lines) {
    std::size_t num_leading_whitespace = 0;
    bool non_space_found = false;
    for (auto c : line) {
      if (c == ' ' || c == '\t') {
        num_leading_whitespace++;
      } else {
        non_space_found = true;
        break;
      }
    }
    if (non_space_found)
      min_indent = std::min(min_indent, num_leading_whitespace);
    if (min_indent == 0)
      break;
  }

  source_code.clear();
  for (auto &line : lines)
    source_code += line.substr(std::min(line.size(), min_indent)) + '\n';

  return min_indent;
}

std::string get_source_code(const py::callable &func) {
  py::module_ analysis = py::module_::import_("cudaq.kernel.analysis");
  py::object FetchDepFuncsSourceCode = analysis.attr("FetchDepFuncsSourceCode");
  py::object source_code;
  try {
    source_code = FetchDepFuncsSourceCode.attr("fetch")(func);
  } catch (py::python_error &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = py::cast<std::string>(source_code);
  strip_leading_whitespace(source);
  return source;
}

std::string get_var_name_for_handle(const py::handle &h) {
  py::object inspect = py::module_::import_("inspect");
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    py::dict f_locals = current_frame.attr("f_locals");
    for (auto item : f_locals)
      if (item.second.is(h))
        return std::string(py::str(item.first).c_str());
    current_frame = current_frame.attr("f_back");
  }
  current_frame = inspect.attr("currentframe")();
  py::dict f_globals = current_frame.attr("f_globals");
  for (auto item : f_globals)
    if (item.second.is(h))
      return std::string(py::str(item.first).c_str());
  return std::string();
}

std::unordered_map<std::string, std::tuple<py::object, py::dict>>
    DataClassRegistry::classes{};

/// @brief Bind the dataclass registry
void bindPyDataClassRegistry(py::module_ &mod) {
  py::class_<DataClassRegistry>(mod, "DataClassRegistry",
                                R"#(Registry for dataclasses used in kernels)#")
      .def_static("registerClass", &DataClassRegistry::registerClass,
                  "Register class\n")
      .def_static("isRegisteredClass", &DataClassRegistry::isRegisteredClass,
                  "Is class registered\n")
      .def_static("getClassAttributes", &DataClassRegistry::getClassAttributes,
                  "Find registered class and its attributes\n")
      .def_static(
          "get_classes",
          []() -> decltype(DataClassRegistry::classes) & {
            return DataClassRegistry::classes;
          },
          py::rv_policy::reference, "Get all registered classes.")
      .def_prop_ro_static(
          "classes",
          [](py::handle /*cls*/) -> decltype(DataClassRegistry::classes) & {
            return DataClassRegistry::classes;
          },
          py::rv_policy::reference, "Get all registered classes.");
}
} // namespace cudaq
