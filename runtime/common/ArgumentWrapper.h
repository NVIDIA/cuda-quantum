/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <vector>

namespace cudaq {
/// The OpaqueArguments type wraps a vector of function arguments represented as
/// opaque pointers. For each element in the vector of opaque pointers, we also
/// track the arguments corresponding deletion function - a function invoked
/// upon destruction of this OpaqueArguments to clean up the memory.
class OpaqueArguments {
public:
  using OpaqueArgDeleter = std::function<void(void *)>;
  OpaqueArguments() = default;
  // No copy constructor is allowed
  // The `OpaqueArguments` instance is the sole owner of its data.
  OpaqueArguments(const OpaqueArguments &) = delete;
  OpaqueArguments &operator=(const OpaqueArguments &) = delete;

  // Move constructor
  OpaqueArguments(OpaqueArguments &&other) noexcept
      : args(std::move(other.args)), deleters(std::move(other.deleters)) {
    other.args.clear();
    other.deleters.clear();
  }

  OpaqueArguments &operator=(OpaqueArguments &&other) noexcept {
    if (this != &other) {
      args = std::move(other.args);
      deleters = std::move(other.deleters);
      other.args.clear();
      other.deleters.clear();
    }
    return *this;
  }

  /// Does \e not make a copy of `args`.
  const std::vector<void *> &getArgs() const { return args; }

  /// @brief Add an opaque argument and its `deleter` to this OpaqueArguments
  template <typename ArgPointer, typename Deleter>
  void emplace_back(ArgPointer &&pointer, Deleter &&deleter) {
    args.emplace_back(pointer);
    deleters.emplace_back(deleter);
  }

  /// @brief Return the number of arguments
  std::size_t size() { return args.size(); }
  bool empty() const { return args.empty(); }

  /// Destructor, clean up the memory
  ~OpaqueArguments() {
    for (std::size_t counter = 0; auto &ptr : args)
      deleters[counter++](ptr);

    args.clear();
    deleters.clear();
  }

private:
  /// @brief The opaque argument pointers
  std::vector<void *> args;

  /// @brief Deletion functions for the arguments.
  std::vector<OpaqueArgDeleter> deleters;
};

namespace runtime {
class CallableClosureArgument {
public:
  explicit CallableClosureArgument(const std::string &name, mlir::ModuleOp mod,
                                   std::optional<unsigned> &&startLifted,
                                   OpaqueArguments &&oppy)
      : short_name{name}, module_op{mod},
        start_lifted_pos{std::move(startLifted)}, opaque_args{std::move(oppy)} {
  }

  const std::string &getShortName() const { return short_name; }
  mlir::ModuleOp getModule() { return module_op; }
  const std::optional<unsigned> &getStartLiftedPos() const {
    return start_lifted_pos;
  }
  const std::vector<void *> &getArgs() const { return opaque_args.getArgs(); }

private:
  CallableClosureArgument() = delete;

  std::string short_name;
  mlir::ModuleOp module_op;
  // Positions of the opaque arguments.
  std::optional<unsigned> start_lifted_pos;
  // Using OpaqueArguments class here since it deals with struct{any}.
  OpaqueArguments opaque_args;
};
} // namespace runtime

// TODO: should the following two classes be deprecated?
struct ArgWrapper {
  mlir::ModuleOp mod;
  std::vector<std::string> callableNames;
  void *rawArgs = nullptr;
};

/// Holder of wrapped kernel `args`.
struct KernelArgsHolder {
  cudaq::ArgWrapper argsWrapper;
  // Info about the argsWrapper's rawArgs pointer.
  std::size_t argsSize;
  std::int32_t returnOffset;
};

} // namespace cudaq
