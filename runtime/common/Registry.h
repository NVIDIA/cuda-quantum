/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1
#include "llvm/Support/Registry.h"
#include <memory>

namespace cudaq {
namespace registry {

/// @brief RegisteredType allows interface types to declare themselves
/// as plugin interfaces. Used as follows
/// class MyInterface : public RegisteredType<MyInterface> {...};
template <typename T>
class RegisteredType {
public:
  using RegistryType = llvm::Registry<T>;
};

/// @brief Retrieve a plugin sub-type of the given template type by name.
template <typename T>
std::unique_ptr<T> get(const std::string &name) {
  for (typename T::RegistryType::iterator it = T::RegistryType::begin(),
                                          ie = T::RegistryType::end();
       it != ie; ++it) {
    if (name == it->getName().str()) {
      return it->instantiate();
    }
  }
  return nullptr;
}

/// @brief Return true if the plugin with given name and type is available.
template <typename T>
bool isRegistered(const std::string &name) {
  for (typename T::RegistryType::iterator it = T::RegistryType::begin(),
                                          ie = T::RegistryType::end();
       it != ie; ++it) {
    if (name == it->getName().str())
      return true;
  }
  return false;
}

} // namespace registry
} // namespace cudaq

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define CUDAQ_REGISTER_TYPE(TYPE, SUBTYPE, NAME)                               \
  static TYPE::RegistryType::Add<SUBTYPE> CONCAT(TMPNAME_, NAME)(#NAME, "");
