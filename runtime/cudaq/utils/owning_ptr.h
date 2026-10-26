/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <memory>

namespace cudaq {
/// @brief A `deleter` to be used with @c owning_ptr to hold a @c
/// std::unique_ptr<T> in a header that may only forward-declares @c T.
///
/// @code
/// Example:
/// // MyType.h
/// class MyType {
///     ...
/// };
///
/// // MyType.cpp
/// // Only provide an implementation of the `deleter` in the same TU as the
/// type. void opaque_deleter<MyType>::operator()(MyType *p) `const` { delete p;
/// }
///
/// // Other header
/// // Forward-declare the type.
/// class MyType;
/// // Use the owning_ptr in the declaration.
/// owning_ptr<MyType> my_ptr;
/// @endcode
///
template <typename T>
struct opaque_deleter {
  void operator()(T *p) const;
};

/// @brief A @c std::unique_ptr<T> whose destruction is performed by an
/// out-of-line @c opaque_deleter<T> specialization.
/// The goal is to provide the same type erasure afforded by an @c
/// std::shared_ptr while retaining the semantics of an @c std::unique_ptr.
template <typename T>
using owning_ptr = std::unique_ptr<T, opaque_deleter<T>>;

} // namespace cudaq
