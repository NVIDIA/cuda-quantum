/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Suppress the LLVM ABI breaking check for translation units that include this
// header. Several server helper DSOs transitively include LLVM headers (e.g.
// llvm/Support/Base64.h) but do not link against LLVM libraries, so the ABI
// check symbol would be unresolved. This define must be set before any LLVM
// header is included. TODO: remove once those DSOs stop including LLVM headers.
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include <memory>
#include <string>

namespace cudaq {

/// A registry entry: name + factory function that constructs a
/// std::unique_ptr<T>. Modeled after llvm::SimpleRegistryEntry.
template <typename T>
class RegistryEntry {
  const char *Name;
  std::unique_ptr<T> (*Ctor)();

public:
  RegistryEntry(const char *N, std::unique_ptr<T> (*C)()) : Name(N), Ctor(C) {}

  const char *getName() const { return Name; }
  std::unique_ptr<T> instantiate() const { return Ctor(); }
};

/// A global type registry used with static constructors to make pluggable
/// components "just work" when linked with an executable or shared library.
///
/// This is a drop-in replacement for llvm::Registry<T> that has no LLVM
/// dependencies. The cross-shared-library mechanism is the same: Head, Tail,
/// and add_node are defined in exactly one translation unit per registry type
/// via CUDAQ_INSTANTIATE_REGISTRY, so all DSOs that link against that TU
/// share a single list.
template <typename T>
class Registry {
public:
  using type = T;
  using entry = RegistryEntry<T>;

  class node;
  class iterator;

private:
  Registry() = delete;

  friend class node;
  static node *Head, *Tail;

public:
  /// Node in the singly-linked list of entries.
  class node {
    friend class iterator;
    friend class Registry<T>;

    node *Next;
    const entry &Val;

  public:
    node(const entry &V) : Next(nullptr), Val(V) {}
  };

  /// Append a node to the list. Defined out-of-line by
  /// CUDAQ_INSTANTIATE_REGISTRY so that the single definition lives in
  /// the DSO that owns this registry.
  static void add_node(node *N);

  /// Forward iterator over registry entries.
  class iterator {
    const node *Cur;

  public:
    using value_type = const entry;
    using reference = const entry &;
    using pointer = const entry *;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    explicit iterator(const node *N) : Cur(N) {}

    bool operator==(const iterator &That) const { return Cur == That.Cur; }
    bool operator!=(const iterator &That) const { return Cur != That.Cur; }
    iterator &operator++() {
      Cur = Cur->Next;
      return *this;
    }
    const entry &operator*() const { return Cur->Val; }
    const entry *operator->() const { return &Cur->Val; }
  };

  /// begin() is defined by CUDAQ_INSTANTIATE_REGISTRY (same pattern as LLVM).
  static iterator begin();
  static iterator end() { return iterator(nullptr); }

  /// Lightweight range for range-based for loops.
  struct range {
    iterator b, e;
    iterator begin() const { return b; }
    iterator end() const { return e; }
  };
  static range entries() { return {begin(), end()}; }

  /// Static registration helper. Constructed as a global/static object to
  /// register a concrete subtype V under a string name:
  ///
  ///   static Registry<Base>::Add<Derived> reg("name");
  template <typename V>
  class Add {
    entry Entry;
    node Node;

    static std::unique_ptr<T> CtorFn() { return std::make_unique<V>(); }

  public:
    Add(const char *Name) : Entry(Name, CtorFn), Node(Entry) {
      add_node(&Node);
    }
  };
};

namespace registry {

/// Mixin base class: inherit from this to declare T as a register-able type.
///   class QPU : public RegisteredType<QPU> { ... };
template <typename T>
class RegisteredType {
public:
  using RegistryType = ::cudaq::Registry<T>;
};

/// Retrieve a registered subtype by name.
template <typename T>
std::unique_ptr<T> get(const std::string &name) {
  for (auto it = T::RegistryType::begin(), ie = T::RegistryType::end();
       it != ie; ++it) {
    if (name == it->getName())
      return it->instantiate();
  }
  return nullptr;
}

/// Return true if a subtype with the given name is registered.
template <typename T>
bool isRegistered(const std::string &name) {
  for (auto it = T::RegistryType::begin(), ie = T::RegistryType::end();
       it != ie; ++it) {
    if (name == it->getName())
      return true;
  }
  return false;
}

} // namespace registry
} // namespace cudaq

/// Instantiate a registry for a single type. Place this in exactly one `.cpp`
/// per registry type in the DSO that should own the list.
///
/// REGISTRY_CLASS is the Registry typedef, e.g. cudaq::QPU::RegistryType.
///
/// This mirrors LLVM_INSTANTIATE_REGISTRY: it provides the template
/// definitions for Head, Tail, add_node, and begin, then forces explicit
/// instantiation for the concrete type.
#define CUDAQ_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                             \
  namespace cudaq {                                                            \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Head = nullptr;                     \
  template <typename T>                                                        \
  typename Registry<T>::node *Registry<T>::Tail = nullptr;                     \
  template <typename T>                                                        \
  void Registry<T>::add_node(typename Registry<T>::node *N) {                  \
    if (Tail)                                                                  \
      Tail->Next = N;                                                          \
    else                                                                       \
      Head = N;                                                                \
    Tail = N;                                                                  \
  }                                                                            \
  template <typename T>                                                        \
  typename Registry<T>::iterator Registry<T>::begin() {                        \
    return iterator(Head);                                                     \
  }                                                                            \
  template                                                                     \
      typename REGISTRY_CLASS::node *Registry<REGISTRY_CLASS::type>::Head;     \
  template                                                                     \
      typename REGISTRY_CLASS::node *Registry<REGISTRY_CLASS::type>::Tail;     \
  template void                                                                \
  Registry<REGISTRY_CLASS::type>::add_node(typename REGISTRY_CLASS::node *);   \
  template typename REGISTRY_CLASS::iterator                                   \
  Registry<REGISTRY_CLASS::type>::begin();                                     \
  }

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define CUDAQ_REGISTER_TYPE(TYPE, SUBTYPE, NAME)                               \
  static TYPE::RegistryType::Add<SUBTYPE> CONCAT(TMPNAME_, NAME)(#NAME);
