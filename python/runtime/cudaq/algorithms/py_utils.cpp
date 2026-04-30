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

nanobind::dict get_serializable_var_dict() {
  nanobind::object json = nanobind::module_::import_("json");
  nanobind::dict serialized_dict;

  auto try_to_add_item = [&](const auto item) {
    try {
      auto key = item.first;
      auto value = item.second;

      std::string keyStr(nanobind::str(key).c_str());
      if (keyStr.starts_with("__")) {
        // Ignore items that start with "__" (like Python __builtins__, etc.)
      } else if (nanobind::hasattr(value, "to_json")) {
        auto type = nanobind::handle(
            reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())));
        std::string module(nanobind::str(type.attr("__module__")).c_str());
        std::string name(nanobind::str(type.attr("__name__")).c_str());
        auto type_name = nanobind::str((module + "." + name).c_str());
        nanobind::str json_key_name(
            (keyStr + "/" + module + "." + name).c_str());
        serialized_dict[json_key_name] =
            json.attr("loads")(value.attr("to_json")());
      } else if (nanobind::hasattr(value, "tolist")) {
        serialized_dict[key] =
            json.attr("loads")(json.attr("dumps")(value.attr("tolist")()));
      } else {
        serialized_dict[key] = json.attr("loads")(json.attr("dumps")(value));
      }
    } catch (const nanobind::python_error &e) {
      // Serialization failures are non-fatal - we just skip the entry.
    }
  };

  for (const auto item : nanobind::globals())
    try_to_add_item(item);

  nanobind::object inspect = nanobind::module_::import_("inspect");
  std::vector<nanobind::object> frame_vec;
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    frame_vec.push_back(nanobind::object(current_frame));
    current_frame = current_frame.attr("f_back");
  }

  // Walk backwards through the call stack, which means we are going from
  // globals first to locals last. This ensures that the overwrites give
  // precedence to closest-to-locals.
  for (auto it = frame_vec.rbegin(); it != frame_vec.rend(); ++it) {
    nanobind::dict f_locals = it->attr("f_locals");
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

std::string get_source_code(const nanobind::callable &func) {
  // Get the source code
  nanobind::module_ analysis =
      nanobind::module_::import_("cudaq.kernel.analysis");
  nanobind::object FetchDepFuncsSourceCode =
      analysis.attr("FetchDepFuncsSourceCode");
  nanobind::object source_code;
  try {
    source_code = FetchDepFuncsSourceCode.attr("fetch")(func);
  } catch (nanobind::python_error &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = nanobind::cast<std::string>(source_code);
  strip_leading_whitespace(source);
  return source;
}

std::string get_var_name_for_handle(const nanobind::handle &h) {
  nanobind::object inspect = nanobind::module_::import_("inspect");
  // Search locals first, walking up the call stack
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    nanobind::dict f_locals = current_frame.attr("f_locals");
    for (auto item : f_locals)
      if (item.second.is(h))
        return std::string(nanobind::str(item.first).c_str());
    current_frame = current_frame.attr("f_back");
  }
  // Search globals now
  current_frame = inspect.attr("currentframe")();
  nanobind::dict f_globals = current_frame.attr("f_globals");
  for (auto item : f_globals)
    if (item.second.is(h))
      return std::string(nanobind::str(item.first).c_str());
  return std::string();
}

std::unordered_map<std::string, std::tuple<nanobind::object, nanobind::dict>>
    DataClassRegistry::classes{};

/// @brief Bind the dataclass registry
void bindPyDataClassRegistry(nanobind::module_ &mod) {
  nanobind::class_<DataClassRegistry>(
      mod, "DataClassRegistry", R"#(Registry for dataclasses used in kernels)#")
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
          nanobind::rv_policy::reference, "Get all registered classes.")
      .def_prop_ro_static(
          "classes",
          [](nanobind::handle /*cls*/)
              -> decltype(DataClassRegistry::classes) & {
            return DataClassRegistry::classes;
          },
          nanobind::rv_policy::reference, "Get all registered classes.");
}
} // namespace cudaq
