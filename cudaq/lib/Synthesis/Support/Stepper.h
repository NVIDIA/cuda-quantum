/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/iterator.h"

#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudaq::synth {

/// CRTP base for lazy, single-pass steppers.
///
/// A stepper is a non-coroutine replacement for the previous C++20
/// `generator<T>` pattern, following plain C++17 and MLIR/LLVM ADT
/// conventions. Derived classes implement
///
///     const T *next();
///
/// returning a pointer to the next value (valid until the next call to
/// `next()` or the stepper's destruction), or `nullptr` when exhausted.
///
/// `StepperBase` adds a single-pass input range interface around `next()` so
/// that derived steppers can be used directly with range-`for`:
///
///     for (const T &v : MyStepper(...)) { ... }
///     auto vec = to_vector(MyStepper(...));
///
/// The iterator type inherits from `llvm::iterator_facade_base`, matching the
/// rest of the LLVM/MLIR codebase. Derived steppers only need to provide
/// `next()`; the iterator/range glue is handled here.
///
/// Yielded reference contract: the pointer returned by `next()` and the
/// reference returned by `*it` are valid until the next call to `next()` /
/// `++it`. Callers must consume or copy the value before advancing.
///
/// Steppers are typically non-copyable and non-movable (they often own
/// `mpfr_t` scratch state with no move semantics). The base class itself is
/// trivially copyable but derived classes can delete copy/move as needed.
template <typename Derived, typename T>
class StepperBase {
public:
  using value_type = T;

  /// Single-pass input iterator over the stepper's `next()` output.
  /// Default-constructed iterators compare equal to any iterator whose
  /// pointer is null (i.e. an exhausted iterator), which is the `end()`
  /// sentinel for the range.
  class iterator
      : public llvm::iterator_facade_base<iterator, std::input_iterator_tag,
                                          const T> {
  public:
    iterator() = default;
    iterator(Derived *s, const T *v) : s_(s), v_(v) {}

    const T &operator*() const { return *v_; }

    iterator &operator++() {
      v_ = s_->next();
      return *this;
    }

    bool operator==(const iterator &other) const { return v_ == other.v_; }

  private:
    Derived *s_ = nullptr;
    const T *v_ = nullptr;
  };

  iterator begin() {
    auto *self = static_cast<Derived *>(this);
    return iterator(self, self->next());
  }

  iterator end() const { return iterator(); }
};

/// Materialize all elements of a lazy range (a stepper or any class with
/// `begin()` / `end()` yielding T) into a `std::vector`.
template <typename Range>
auto to_vector(Range &&r) {
  using ValueT =
      std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>;
  std::vector<ValueT> result;
  for (const auto &val : r)
    result.push_back(val);
  return result;
}

/// Get the first element from a lazy range, or `std::nullopt` if empty.
template <typename Range>
auto first_of(Range &&r)
    -> std::optional<
        std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>> {
  auto it = r.begin();
  if (it != r.end())
    return *it;
  return std::nullopt;
}

} // namespace cudaq::synth
