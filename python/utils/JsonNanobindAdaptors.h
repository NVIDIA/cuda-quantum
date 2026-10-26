/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "nlohmann/json.hpp"

namespace nanobind {
namespace detail {

/// Bidirectional type caster between nlohmann::json and Python objects.
/// JSON object  <-> dict, array <-> list, string <-> `str`,
/// integer <-> int, float <-> float, bool <-> bool, null <-> None.
template <>
struct type_caster<nlohmann::json> {
  NB_TYPE_CASTER(nlohmann::json, const_name("object"))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      value = python_to_json(src);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(const nlohmann::json &j, rv_policy,
                         cleanup_list *) noexcept {
    try {
      return json_to_python(j).release();
    } catch (...) {
      return handle();
    }
  }

private:
  static nlohmann::json python_to_json(handle src) {
    if (src.is_none())
      return nullptr;
    if (PyBool_Check(src.ptr()))
      return src.equal(nanobind::bool_(true));
    if (PyLong_Check(src.ptr()))
      return nanobind::cast<int64_t>(src);
    if (PyFloat_Check(src.ptr()))
      return nanobind::cast<double>(src);
    if (PyUnicode_Check(src.ptr()))
      return nanobind::cast<std::string>(src);
    if (PyDict_Check(src.ptr())) {
      nlohmann::json obj = nlohmann::json::object();
      for (auto [k, v] : nanobind::borrow<nanobind::dict>(src))
        obj[nanobind::cast<std::string>(k)] = python_to_json(v);
      return obj;
    }
    if (PyList_Check(src.ptr()) || PyTuple_Check(src.ptr())) {
      nlohmann::json arr = nlohmann::json::array();
      for (auto item : nanobind::borrow<nanobind::sequence>(src))
        arr.push_back(python_to_json(item));
      return arr;
    }
    throw nanobind::type_error("nlohmann::json type caster: unsupported type");
  }

  static nanobind::object json_to_python(const nlohmann::json &j) {
    switch (j.type()) {
    case nlohmann::json::value_t::null:
      return nanobind::none();
    case nlohmann::json::value_t::boolean:
      return nanobind::bool_(j.get<bool>());
    case nlohmann::json::value_t::number_integer:
    case nlohmann::json::value_t::number_unsigned:
      return nanobind::int_(j.get<int64_t>());
    case nlohmann::json::value_t::number_float:
      return nanobind::float_(j.get<double>());
    case nlohmann::json::value_t::string: {
      const auto &s = j.get_ref<const std::string &>();
      return nanobind::str(s.data(), s.size());
    }
    case nlohmann::json::value_t::object: {
      nanobind::dict d;
      for (auto &[k, v] : j.items())
        d[nanobind::str(k.data(), k.size())] = json_to_python(v);
      return d;
    }
    case nlohmann::json::value_t::array: {
      nanobind::list lst;
      for (auto &item : j)
        lst.append(json_to_python(item));
      return lst;
    }
    default:
      return nanobind::none();
    }
  }
};

} // namespace detail
} // namespace nanobind
