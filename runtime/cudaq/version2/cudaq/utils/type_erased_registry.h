/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <memory>
#include <unordered_map>

#pragma once

namespace cudaq {
namespace registry {
/// The Registry Singleton
///
/// This class maintains a global registry of available implementations
/// for wrapper type any_type.
/// A dispatch table must be provided to create the object and build a
/// type-erased manual virtual table.
///
/// Example for any_qpu
///
/// struct qpu_dispatch_table {
///   std::function<std::shared_ptr<void>()> create;
///   std::function<void(void *, std::size_t)> setId;
///
///   template <typename T>
///   void build() {
///     create = []() { return std::make_shared<T>(); };
///     setId = [](void *i, std::size_t id) { static_cast<T *>(i)->setId(id); };
///   }
/// };
///
/// class any_qpu {
///   std::shared_ptr<void> m_instance;
///   qpu_dispatch_table const *const m_vtable;
///
/// public:
///   any_qpu(std::shared_ptr<void> instance,
///           const details::qpu_dispatch_table *vtable)
///       : m_instance(instance), m_vtable(vtable) {}
///   void setId(std::size_t id) { m_vtable->setId(m_instance.get(), id); }
/// };
///
template <typename any_type, typename dispatch_table>
class TypeErasedRegistry {
public:
  static TypeErasedRegistry &get();

  template <typename T>
  void register_type(const std::string &name) {
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

/// A registrar helper to register types at static initialization time
///
/// Example for any_qpu:
///
/// using QPURegistry =
///     TypeErasedRegistry<any_qpu, cudaq::details::qpu_dispatch_table>;
///
/// #define CUDAQ_REGISTER_QPU_TYPE(TYPE, NAME) \
///   static TypeErasedRegistrar<cudaq::registry::QPURegistry, TYPE> CONCAT( \
///       registrar, NAME)(#NAME);
template <typename Registry, typename actual_type>
class TypeErasedRegistrar {
public:
  TypeErasedRegistrar(const std::string &name) {
    Registry::get().template register_type<actual_type>(name);
  }
};

} // namespace registry
} // namespace cudaq

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define CUDAQ_INSTANTIATE_TYPE_ERASED_REGISTRY(REGISTRY)                       \
  template <>                                                                  \
  REGISTRY &REGISTRY::get() {                                                  \
    static REGISTRY instance;                                                  \
    return instance;                                                           \
  }
