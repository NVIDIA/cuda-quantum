/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace cudaq {

/// A handle to an item in a contiguously stored container, e.g., vector or
/// array.
///
/// This class is designed to be lightweight and thus values of this type
/// should be passed by value, _not_ reference or pointer.
struct Handle {
  static constexpr unsigned InvalidIndex = std::numeric_limits<unsigned>::max();

  static constexpr Handle getInvalid() { return Handle(InvalidIndex); }

  Handle() : index(InvalidIndex) {}
  constexpr explicit Handle(unsigned index) : index(index) {}

  LLVM_DUMP_METHOD void print(llvm::raw_ostream &os) const {
    if (isValid())
      os << index;
    else
      os << "<invalid>";
  }

  bool isValid() const { return index != InvalidIndex; }

  unsigned index;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Handle handle) {
  handle.print(os);
  return os;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator==(HandleTy lhs, HandleTy rhs) {
  return lhs.index == rhs.index;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator!=(HandleTy lhs, HandleTy rhs) {
  return lhs.index != rhs.index;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator<(HandleTy lhs, HandleTy rhs) {
  return lhs.index < rhs.index;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator<=(HandleTy lhs, HandleTy rhs) {
  return lhs.index <= rhs.index;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator>(HandleTy lhs, HandleTy rhs) {
  return lhs.index > rhs.index;
}

template <
    typename HandleTy,
    typename std::enable_if_t<std::is_base_of_v<Handle, HandleTy>> * = nullptr>
bool operator>=(HandleTy lhs, HandleTy rhs) {
  return lhs.index >= rhs.index;
}

} // namespace cudaq
