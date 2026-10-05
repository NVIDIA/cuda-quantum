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
