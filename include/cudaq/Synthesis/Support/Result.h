/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// LogicalResult: lightweight success/failure indicator
// ---------------------------------------------------------------------------
//
// Modeled on mlir::LogicalResult. Represents a binary outcome with no
// payload. Use for functions that perform an action and can fail.
//
// Usage:
//   LogicalResult doSomething() {
//     if (bad) return failure();
//     return success();
//   }
//
//   if (failed(doSomething())) handleError();
//
class [[nodiscard]] LogicalResult {
public:
  static LogicalResult success() { return LogicalResult(false); }
  static LogicalResult failure() { return LogicalResult(true); }

  bool isFailure() const noexcept { return failed_; }
  bool isSuccess() const noexcept { return !failed_; }

  // Prevent accidental if (result) — force succeeded()/failed() helpers.
  explicit operator bool() const = delete;

private:
  explicit LogicalResult(bool failed) : failed_(failed) {}
  bool failed_;
};

inline LogicalResult success() { return LogicalResult::success(); }
inline LogicalResult failure() { return LogicalResult::failure(); }
inline bool succeeded(LogicalResult r) noexcept { return r.isSuccess(); }
inline bool failed(LogicalResult r) noexcept { return r.isFailure(); }

// ---------------------------------------------------------------------------
// FailureOr<T>: either a T value or a failure
// ---------------------------------------------------------------------------
//
// Modeled on mlir::FailureOr<T>. Holds either a successfully computed T or
// a failure state. Accessing the value without first checking triggers an
// assertion failure in debug builds; in release it is undefined behaviour,
// so callers must always guard with failed()/succeeded() before operator*.
//
// Usage:
//   FailureOr<Foo> makeFoo(...) {
//     if (bad) return failure();
//     return Foo(...);          // implicit T→FailureOr<T> conversion
//   }
//
//   auto r = makeFoo(...);
//   if (failed(r)) return failure();
//   Foo &foo = *r;
//
template <typename T>
class [[nodiscard]] FailureOr {
  static_assert(!std::is_same_v<std::decay_t<T>, LogicalResult>,
                "Use LogicalResult directly, not FailureOr<LogicalResult>");

public:
  FailureOr() : storage_(FailureTag{}) {}

  // Implicit conversion from LogicalResult (must be a failure).
  /* implicit */ FailureOr(LogicalResult r) : storage_(FailureTag{}) {
    assert(r.isFailure() &&
           "FailureOr<T> can only be constructed from a failure LogicalResult");
  }

  // Implicit conversion from a successfully produced value.
  /* implicit */ FailureOr(T &&value)
      : storage_(std::in_place_index<1>, std::move(value)) {}
  /* implicit */ FailureOr(const T &value)
      : storage_(std::in_place_index<1>, value) {}

  bool isFailure() const noexcept { return storage_.index() == 0; }
  bool isSuccess() const noexcept { return storage_.index() == 1; }

  T &operator*() {
    assert(isSuccess() && "Dereferencing a failed FailureOr<T>");
    return std::get<1>(storage_);
  }
  const T &operator*() const {
    assert(isSuccess() && "Dereferencing a failed FailureOr<T>");
    return std::get<1>(storage_);
  }

  T *operator->() {
    assert(isSuccess() && "Dereferencing a failed FailureOr<T>");
    return &std::get<1>(storage_);
  }
  const T *operator->() const {
    assert(isSuccess() && "Dereferencing a failed FailureOr<T>");
    return &std::get<1>(storage_);
  }

  // Prevent accidental if (result) — force succeeded()/failed() helpers.
  explicit operator bool() const = delete;

private:
  struct FailureTag {};
  std::variant<FailureTag, T> storage_;
};

template <typename T>
bool succeeded(const FailureOr<T> &r) noexcept {
  return r.isSuccess();
}

template <typename T>
bool failed(const FailureOr<T> &r) noexcept {
  return r.isFailure();
}

// ---------------------------------------------------------------------------
// DiagnosticResult / DiagnosticOr<T>: success/failure with an error message
// ---------------------------------------------------------------------------
//
// Use at API boundaries or diagnostic-worthy error paths where an error
// message adds value. Internal code should prefer LogicalResult / FailureOr<T>
// for zero overhead.
//
class [[nodiscard]] DiagnosticResult {
public:
  static DiagnosticResult success() { return DiagnosticResult(std::nullopt); }
  static DiagnosticResult failure(std::string msg) {
    return DiagnosticResult(std::optional<std::string>(std::move(msg)));
  }

  bool isFailure() const noexcept { return message_.has_value(); }
  bool isSuccess() const noexcept { return !message_.has_value(); }

  const std::string &message() const {
    assert(isFailure() && "No diagnostic message on a successful result");
    return *message_;
  }

  explicit operator bool() const = delete;

private:
  explicit DiagnosticResult(std::optional<std::string> msg)
      : message_(std::move(msg)) {}
  std::optional<std::string> message_;
};

inline bool succeeded(const DiagnosticResult &r) noexcept {
  return r.isSuccess();
}
inline bool failed(const DiagnosticResult &r) noexcept { return r.isFailure(); }

template <typename T>
class [[nodiscard]] DiagnosticOr {
public:
  /* implicit */ DiagnosticOr(T &&value)
      : storage_(std::in_place_index<1>, std::move(value)) {}
  /* implicit */ DiagnosticOr(const T &value)
      : storage_(std::in_place_index<1>, value) {}

  static DiagnosticOr failure(std::string msg) {
    DiagnosticOr result;
    std::get<0>(result.storage_) = std::move(msg);
    return result;
  }

  bool isFailure() const noexcept { return storage_.index() == 0; }
  bool isSuccess() const noexcept { return storage_.index() == 1; }

  const std::string &message() const {
    assert(isFailure() && "No diagnostic message on a successful result");
    return std::get<0>(storage_);
  }

  T &operator*() {
    assert(isSuccess() && "Dereferencing a failed DiagnosticOr<T>");
    return std::get<1>(storage_);
  }
  const T &operator*() const {
    assert(isSuccess() && "Dereferencing a failed DiagnosticOr<T>");
    return std::get<1>(storage_);
  }

  explicit operator bool() const = delete;

private:
  DiagnosticOr() : storage_(std::string{}) {}
  std::variant<std::string, T> storage_;
};

template <typename T>
bool succeeded(const DiagnosticOr<T> &r) noexcept {
  return r.isSuccess();
}
template <typename T>
bool failed(const DiagnosticOr<T> &r) noexcept {
  return r.isFailure();
}

} // namespace cudaq::synth
