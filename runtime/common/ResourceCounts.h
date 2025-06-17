/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {
/// Type definition for the mapping of observed qubit measurement bit strings
/// to the number of times they were observed.
using CountsDictionary = std::unordered_map<std::string, std::size_t>;

class resource_counts {
private:
  CountsDictionary gateCounts;

  /// @brief Keep track of the total number of gates. We keep this
  /// here so we don't have to keep recomputing it.
  std::size_t totalGates = 0;

  std::size_t numQubits = 0;

  std::string trace = "";

public:
  struct GateData {
    std::string name;
    size_t controls;
  };

  /// @brief Nullary constructor
  resource_counts() = default;

  /// @brief Copy Constructor
  resource_counts(const resource_counts &) = default;

  /// @brief Move constructor
  resource_counts(resource_counts &&) = default;

  /// @brief Move assignment constructor
  resource_counts &operator=(resource_counts &&counts) = default;

  /// @brief The destructor
  ~resource_counts() = default;

  /// @brief Add another gate to this `resource_count`.
  /// @param gate Encountered gate
  void append(const GateData &gate, size_t count = 1);

  /// @brief Add another gate to this `resource_count`.
  /// @param gate Encountered gate
  void append(const std::string &gate, size_t count = 1);

  /// @brief Return the number of times the given gate was observed
  /// @param gate
  /// @return
  std::size_t count(const GateData &gate) const;

  /// @brief Return the number of times the given gate was observed
  /// @param gate
  /// @return
  std::size_t count(const std::string &gate) const;

  /// @brief Dump this resource_counts to standard out.
  void dump() const;

  /// @brief Dump this resource_counts to the given output stream
  void dump(std::ostream &os) const;

  /// @brief Clear this resource_counts.
  void clear();

  CountsDictionary to_map() const;

  void addQubit();
};

} // namespace cudaq
