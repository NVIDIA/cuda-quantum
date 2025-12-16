/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <functional>
#include <memory>
#include <unordered_map>

#pragma once

template <typename any_type, typename dispatch_table> class TypeErasedRegistry {
public:
  static TypeErasedRegistry &get() {
    static TypeErasedRegistry i;
    return i;
  }

  template <typename T> void register_type(const std::string &name) {
    auto t = dispatch_table();
    t.template build<T>();
    m_dispatch_tables.emplace(name, t);
  }

  std::unique_ptr<any_type> instantiate(const std::string &name) {
    auto it = m_dispatch_tables.find(name);
    if (it == m_dispatch_tables.end())
      return nullptr;

    return std::make_unique<any_type>(it->second.create(), &it->second);
  }

  bool is_registered(const std::string &name) {
    return m_dispatch_tables.find(name) != m_dispatch_tables.end();
  }
private:
  std::unordered_map<std::string, dispatch_table> m_dispatch_tables;
  TypeErasedRegistry() = default;
};

template <typename Registry, typename actual_type> class TypeErasedRegistrar {
public:
  TypeErasedRegistrar(const std::string &name) {
    Registry::get().template register_type<actual_type>(name);
  }
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
