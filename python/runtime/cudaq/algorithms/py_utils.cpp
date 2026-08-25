/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_utils.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>

namespace cudaq {

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
