/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");

  py::dict serialized_dict;
  for (const auto item : py::globals()) {
    try {
      auto key = item.first;
      auto value = item.second;

      py::bytes serialized_value = pickle.attr("dumps")(value);
      serialized_dict[key] = serialized_value;
    } catch (const py::error_already_set &e) {
      // Uncomment the following lines for debug, but all this really means is
      // that we won't send this to the remote server.

      // std::cout << "Failed to pickle key: " + std::string(e.what())
      //           << std::endl;
    }
  }

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
    for (const auto item : f_locals) {
      try {
        auto key = item.first;
        auto value = item.second;

        py::bytes serialized_value = pickle.attr("dumps")(value);
        serialized_dict[key] = serialized_value;
      } catch (const py::error_already_set &e) {
        // Uncomment the following lines for debug, but all this really means is
        // that we won't send this to the remote server.

        // std::cout << "Failed to pickle key: " + std::string(e.what())
        //           << std::endl;
      }
    }
  }

  return serialized_dict;
}

std::string b64encode_dict(py::dict serializable_dict) {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");
  py::bytes serialized_code = pickle.attr("dumps")(serializable_dict);
  py::object encoded_dict = base64.attr("b64encode")(serialized_code);
  return encoded_dict.cast<std::string>();
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
  py::object inspect = py::module::import("inspect");
  py::object source_code;
  try {
    source_code = inspect.attr("getsource")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = source_code.cast<std::string>();
  strip_leading_whitespace(source);
  return source;
}

std::string get_imports() {
  py::module sys = py::module::import("sys");
  py::dict sys_modules = sys.attr("modules");
  py::dict globals = py::globals();
  std::string imports_str;

  for (auto item : globals) {
    if (py::isinstance<py::module>(item.second)) {
      py::module mod = item.second.cast<py::module>();
      std::string alias = py::str(item.first);
      std::string name = py::str(mod.attr("__name__"));
      if (alias == name)
        imports_str += "import " + name + "\n";
      else if (!alias.starts_with("@"))
        imports_str += "import " + name + " as " + alias + "\n";
    }
  }
  return imports_str;
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

} // namespace cudaq
