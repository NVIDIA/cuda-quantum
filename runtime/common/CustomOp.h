/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {
/// @brief Define a `unitary_operation` type that exposes
/// a sub-type specific unitary representation of the
/// operation.
struct unitary_operation {
  /// @brief Given a set of rotation parameters, return
  /// a row-major 1D array representing the unitary operation
  virtual std::vector<std::complex<double>> unitary(
      const std::vector<double> &parameters = std::vector<double>()) const = 0;
  virtual ~unitary_operation() {}
};

/// @brief Singleton class for managing and storing unitary operations.
class customOpRegistry {
public:
  /// @brief Get the singleton instance of the `customOpRegistry`.
  static customOpRegistry &getInstance();

private:
  /// @brief Constructor
  // Private to prevent direct instantiation.
  customOpRegistry() {}

public:
  customOpRegistry(const customOpRegistry &) = delete;
  void operator=(const customOpRegistry &) = delete;

  /// @brief Register a new custom unitary operation under the
  /// provided operation name.
  template <typename T>
  void registerOperation(const std::string &name) {
    {
      std::shared_lock<std::shared_mutex> lock(mtx);
      auto iter = registeredOperations.find(name);
      if (iter != registeredOperations.end())
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mtx);
    registeredOperations.insert({name, std::make_unique<T>()});
  }

  /// Clear the registered operations
  void clearRegisteredOperations();

  /// Returns true if the operation with the given name is registered.
  bool isOperationRegistered(const std::string &name);

  /// Get the unitary operation associated with the given name.
  /// This will throw an exception if the operation is not registered.
  const unitary_operation &getOperation(const std::string &name);

private:
  /// @brief Keep track of a registry of user-provided unitary operations.
  std::unordered_map<std::string, std::unique_ptr<cudaq::unitary_operation>>
      registeredOperations;
  /// @brief  Mutex to protect concurrent access to the registry.
  std::shared_mutex mtx;
};
} // namespace cudaq
