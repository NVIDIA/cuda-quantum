/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022-2025 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <any>
#include <string>
#include <unordered_map>

namespace cudaq {

using observer_data = std::unordered_map<std::string, std::any>;

/// @brief The GlobalStateObserver provides an abstract
/// observer pattern for modification of global state
/// required across CUDA-Q libraries, in an effort to
/// decouple direct linkages. Implementations of this
/// interface know when clients change specific global state
/// (provided with a specified key) and can act accordingly.
/// Implementations can also implement a mechanism for responding
/// with new data based on some changed observed state.
class GlobalStateObserver {
public:
  /// @brief Enumerate some known keys we'll require
  /// here so that we define these strings in one place
  struct KnownDataKeys {
    static constexpr const char RandomSeed[] = "random-seed";
    static constexpr const char TearDownMPI[] = "tear-down-mpi";
    static constexpr const char SimulationState[] = "simulation-state";
    static constexpr const char IsSinglePrecision[] = "is-single-precision";
  };

  /// @brief Perform subtype specific actions based on the
  /// changed input state, provided as a map of string keys to
  /// any type.
  virtual void oneWayNotify(const observer_data &data) = 0;

  /// @brief Perform subtype specific actions based on the
  /// changed input data, and return any relevant response.
  virtual std::tuple<bool, observer_data>
  notifyWithResponse(const observer_data &input) = 0;
};

/// @brief Register this GlobalStateObserver so that
/// any future notifications are sent to it.
void registerObserver(GlobalStateObserver *);

/// @brief Notify all observers of the given data modification.
void notifyAll(const observer_data &data);

/// @brief Notify all observers of the given data input,
/// and respond with any relevant data. The first observer to successfully
/// respond will get its data returned.
observer_data notifyWithResponse(const observer_data &data);

} // namespace cudaq
