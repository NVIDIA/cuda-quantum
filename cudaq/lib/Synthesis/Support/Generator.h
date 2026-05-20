/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <coroutine>
#include <iterator>
#include <optional>
#include <vector>

namespace cudaq::synth {

/// Lazy, single-pass generator `coroutine`.
///
/// Key properties:
///   - Pointer semantics: yields store a T* in the promise, avoiding copies
///     of GMP/MPFR-backed types (Integer, ZSqrt2, DSqrt2, DOmega).
///   - RAII: destroying the generator destroys the `coroutine` frame,
///     running `destructors` for all locals (EnumerationScratch, Integer,
///     etc.). Early termination (abandoning before exhaustion) is safe.
///   - Move-only, non-copyable.
///
/// Usage:
///   generator<int> iota(int n) {
///       for (int i = 0; i < n; ++i)
///           co_yield i;
///   }
///   for (int x : iota(10)) { ... }
///
/// Yielded reference contract: *it is valid until the next ++it.
/// Callers must consume or copy the value before advancing.
template <typename T>
class generator {
public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    const T *value_ptr = nullptr;

    generator get_return_object() {
      return generator{handle_type::from_promise(*this)};
    }

    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() { throw; }

    std::suspend_always yield_value(const T &val) noexcept {
      value_ptr = std::addressof(val);
      return {};
    }

    std::suspend_always yield_value(T &&val) noexcept {
      value_ptr = std::addressof(val);
      return {};
    }
  };

  // -- Iterator --
  struct sentinel {};

  struct iterator {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = const T &;
    using pointer = const T *;

    handle_type handle_ = nullptr;

    iterator() = default;
    explicit iterator(handle_type h) : handle_(h) {}

    reference operator*() const noexcept {
      assert(handle_ && !handle_.done());
      return *handle_.promise().value_ptr;
    }

    pointer operator->() const noexcept { return handle_.promise().value_ptr; }

    iterator &operator++() {
      assert(handle_ && !handle_.done());
      handle_.resume();
      return *this;
    }

    void operator++(int) { ++*this; }

    friend bool operator==(const iterator &it, sentinel) noexcept {
      return !it.handle_ || it.handle_.done();
    }
    friend bool operator!=(const iterator &it, sentinel s) noexcept {
      return !(it == s);
    }
    friend bool operator==(sentinel s, const iterator &it) noexcept {
      return it == s;
    }
    friend bool operator!=(sentinel s, const iterator &it) noexcept {
      return !(it == s);
    }
  };

  // -- Generator interface --
  iterator begin() {
    if (handle_)
      handle_.resume();
    return iterator{handle_};
  }

  sentinel end() const noexcept { return {}; }

  // -- Lifecycle --
  generator() = default;
  explicit generator(handle_type h) : handle_(h) {}

  generator(generator &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  generator &operator=(generator &&other) noexcept {
    if (this != &other) {
      if (handle_)
        handle_.destroy();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  generator(const generator &) = delete;
  generator &operator=(const generator &) = delete;

  ~generator() {
    if (handle_)
      handle_.destroy();
  }

  explicit operator bool() const noexcept { return handle_ && !handle_.done(); }

private:
  handle_type handle_ = nullptr;
};

/// Materialize all elements of a lazy range (generator<T> or any class with
/// begin()/end() yielding T) into a vector.
template <typename Range>
auto to_vector(Range &&r) {
  using ValueT = std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>;
  std::vector<ValueT> result;
  for (const auto &val : r)
    result.push_back(val);
  return result;
}

/// Get the first element from a lazy range, or `nullopt` if empty.
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
