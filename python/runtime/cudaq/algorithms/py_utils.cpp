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
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace cudaq {

nb::dict get_serializable_var_dict() {
  nb::object json = nb::module_::import_("json");
  nb::dict serialized_dict;

  auto try_to_add_item = [&](const auto item) {
    try {
      auto key = item.first;
      auto value = item.second;

      if (nb::cast<std::string>(key).starts_with("__")) {
        // Ignore items that start with "__" (like Python __builtins__, etc.)
      } else if (nb::hasattr(value, "to_json")) {
        auto type = value.type();
        std::string module = nb::cast<std::string>(type.attr("__module__"));
        std::string name = nb::cast<std::string>(type.attr("__name__"));
        auto type_name = nb::str((module + "." + name).c_str());
        auto json_key_name =
            nb::str(nb::str(key).c_str()) + nb::str("/") + type_name;
        serialized_dict[json_key_name] =
            json.attr("loads")(value.attr("to_json")());
      } else if (nb::hasattr(value, "tolist")) {
        serialized_dict[key] =
            json.attr("loads")(json.attr("dumps")(value.attr("tolist")()));
      } else {
        serialized_dict[key] = json.attr("loads")(json.attr("dumps")(value));
      }
    } catch (const nb::python_error &e) {
      // Uncomment the following lines for debug, but all this really means is
      // that we won't send this to the remote server.

      // std::cout << "Failed to serialize key '"
      //           << nb::cast<std::string>(item.first)
      //           << "' : " + std::string(e.what()) << std::endl;
    }
  };

  for (const auto item : nb::globals())
    try_to_add_item(item);

  nb::object inspect = nb::module_::import_("inspect");
  std::vector<nb::object> frame_vec;
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    frame_vec.push_back(current_frame);
    current_frame = current_frame.attr("f_back");
  }

  // Walk backwards through the call stack, which means we are going from
  // globals first to locals last. This ensures that the overwrites give
  // precedence to closest-to-locals.
  for (auto it = frame_vec.rbegin(); it != frame_vec.rend(); ++it) {
    nb::dict f_locals = nb::cast<nb::dict>(it->attr("f_locals"));
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

std::string get_source_code(const nb::callable &func) {
  // Get the source code
  nb::module_ analysis = nb::module_::import_("cudaq.kernel.analysis");
  nb::object FetchDepFuncsSourceCode = analysis.attr("FetchDepFuncsSourceCode");
  nb::object source_code;
  try {
    source_code = FetchDepFuncsSourceCode.attr("fetch")(func);
  } catch (nb::python_error &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = nb::cast<std::string>(source_code);
  strip_leading_whitespace(source);
  return source;
}

std::string get_var_name_for_handle(const nb::handle &h) {
  nb::object inspect = nb::module_::import_("inspect");
  // Search locals first, walking up the call stack
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    nb::dict f_locals = nb::cast<nb::dict>(current_frame.attr("f_locals"));
    for (auto item : f_locals)
      if (item.second.is(h))
        return nb::cast<std::string>(nb::str(item.first));
    current_frame = current_frame.attr("f_back");
  }
  // Search globals now
  current_frame = inspect.attr("currentframe")();
  nb::dict f_globals = nb::cast<nb::dict>(current_frame.attr("f_globals"));
  for (auto item : f_globals)
    if (item.second.is(h))
      return nb::cast<std::string>(nb::str(item.first));
  return std::string();
}

std::unordered_map<std::string, std::tuple<nb::object, nb::dict>>
    DataClassRegistry::classes{};

/// @brief Bind the dataclass registry
void bindPyDataClassRegistry(nb::module_ &mod) {
  nb::class_<DataClassRegistry>(mod, "DataClassRegistry",
                                R"#(Registry for dataclasses used in kernels)#")
      .def_static("registerClass", &DataClassRegistry::registerClass,
                  "Register class\n")
      .def_static("isRegisteredClass", &DataClassRegistry::isRegisteredClass,
                  "Is class registered\n")
      .def_static("getClassAttributes", &DataClassRegistry::getClassAttributes,
                  "Find registered class and its attributes\n")
      .def_ro_static("classes", &DataClassRegistry::classes);
}
} // namespace cudaq
