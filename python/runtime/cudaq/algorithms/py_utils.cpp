/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_utils.h"
#include "cudaq/utils/cudaq_utils.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace cudaq {

py::dict get_serializable_var_dict() {
  py::object json = py::module_::import("json");
  py::dict serialized_dict;

  auto try_to_add_item = [&](const auto item) {
    try {
      auto key = item.first;
      auto value = item.second;

      if (key.template cast<std::string>().starts_with("__")) {
        // Ignore items that start with "__" (like Python __builtins__, etc.)
      } else if (py::hasattr(value, "to_json")) {
        auto type = value.get_type();
        std::string module =
            type.attr("__module__").template cast<std::string>();
        std::string name = type.attr("__name__").template cast<std::string>();
        auto type_name = py::str(module + "." + name);
        auto json_key_name = py::str(key) + py::str("/") + type_name;
        serialized_dict[json_key_name] =
            json.attr("loads")(value.attr("to_json")());
      } else if (py::hasattr(value, "tolist")) {
        serialized_dict[key] =
            json.attr("loads")(json.attr("dumps")(value.attr("tolist")()));
      } else {
        serialized_dict[key] = json.attr("loads")(json.attr("dumps")(value));
      }
    } catch (const py::error_already_set &e) {
      // Uncomment the following lines for debug, but all this really means is
      // that we won't send this to the remote server.

      // std::cout << "Failed to serialize key '"
      //           << item.first.template cast<std::string>()
      //           << "' : " + std::string(e.what()) << std::endl;
    }
  };

  for (const auto item : py::globals())
    try_to_add_item(item);

  py::object inspect = py::module::import("inspect");
  std::vector<py::object> frame_vec;
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    frame_vec.push_back(current_frame);
    current_frame = current_frame.attr("f_back");
  }

  // Walk backwards through the call stack, which means we are going from
  // globals first to locals last. This ensures that the overwrites give
  // precedence to closest-to-locals.
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

  // Traverse the lines to calculate min_indent.
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

  // Now strip the leading indentation off the lines.
  source_code.clear();
  for (auto &line : lines)
    source_code += line.substr(std::min(line.size(), min_indent)) + '\n';

  return min_indent;
}

std::string get_source_code(const py::function &func) {
  // Get the source code
  py::module_ analysis = py::module_::import("cudaq.kernel.analysis");
  py::object FetchDepFuncsSourceCode = analysis.attr("FetchDepFuncsSourceCode");
  py::object source_code;
  try {
    source_code = FetchDepFuncsSourceCode.attr("fetch")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = source_code.cast<std::string>();
  strip_leading_whitespace(source);
  return source;
}

std::string get_var_name_for_handle(const py::handle &h) {
  py::object inspect = py::module::import("inspect");
  // Search locals first, walking up the call stack
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    py::dict f_locals = current_frame.attr("f_locals");
    for (auto item : f_locals)
      if (item.second.is(h))
        return py::str(item.first);
    current_frame = current_frame.attr("f_back");
  }
  // Search globals now
  current_frame = inspect.attr("currentframe")();
  py::dict f_globals = current_frame.attr("f_globals");
  for (auto item : f_globals)
    if (item.second.is(h))
      return py::str(item.first);
  return std::string();
}

std::unordered_map<std::string, std::tuple<py::object, py::dict>>
    DataClassRegistry::classes{};

/// @brief Bind the dataclass registry
void bindPyDataClassRegistry(py::module &mod) {
  py::class_<DataClassRegistry>(mod, "DataClassRegistry",
                                R"#(Registry for dataclasses used in kernels)#")
      .def_static("registerClass", &DataClassRegistry::registerClass,
                  "Register class\n")
      .def_static("isRegisteredClass", &DataClassRegistry::isRegisteredClass,
                  "Is class registered\n")
      .def_static("getClassAttributes", &DataClassRegistry::getClassAttributes,
                  "Find registered class and its attributes\n")
      .def_readonly_static("classes", &DataClassRegistry::classes);
}
} // namespace cudaq
