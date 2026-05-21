/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

//===----------------------------------------------------------------------===//
// StepperBase
//===----------------------------------------------------------------------===//

/// CRTP base for lazy single-pass steppers
///
/// Contract for derived classes: implement
///
///     `const` T *next();
///
/// which returns a pointer to the next value (valid until the next call to
/// next() or the stepper's destruction) or `nullptr` when exhausted.
///
/// StepperBase wraps next() in a single-pass input range so the derived
/// stepper can be used directly with range-for and `to_vector`:
///
///     for (`const` T &v : MyStepper(...)) { ... }
///     auto `vec` = to_vector(MyStepper(...));
///
/// Pointer contract: the value returned by next() (and the reference
/// returned by *it) is valid only until the next call to next() / ++it.
/// Callers must consume or copy before advancing.
///
/// Steppers are typically non-copyable and non-movable. The base class itself
/// is trivially copyable; derived classes can delete copy/move as needed.
template <typename Derived, typename T>
class StepperBase {
public:
  using value_type = T;

  /// Single-pass input iterator over the stepper's next() output. A
  /// default-constructed iterator compares equal to any iterator whose
  /// internal value pointer is null, which is what end() returns and what
  /// an exhausted iterator becomes.
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

//===----------------------------------------------------------------------===//
// Range helpers
//===----------------------------------------------------------------------===//

/// Drain a lazy range (a stepper or anything with begin()/end()) into a
/// std::vector. The value type is deduced from the `iterator`'s `dereference`
/// type with `cv`/ref qualifiers stripped.
template <typename Range>
auto to_vector(Range &&r) {
  using ValueT =
      std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>;
  std::vector<ValueT> result;
  for (const auto &val : r)
    result.push_back(val);
  return result;
}

/// First element of a lazy range, or std::nullopt if the range is empty.
template <typename Range>
auto first_of(Range &&r) -> std::optional<
    std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>> {
  auto it = r.begin();
  if (it != r.end())
    return *it;
  return std::nullopt;
}

} // namespace cudaq::synth
