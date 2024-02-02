/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "common/SimulationState.h"

namespace cudaq {

/// @brief The cudaq::state encapsulate backend simulation state vector or
/// density matrix data.
class state {

private:
  /// @brief Reference to the simulation data
  std::shared_ptr<SimulationState> internal;

public:
  /// @brief The constructor, takes the simulation data and owns it
  state(SimulationState *ptrToOwn)
      : internal(std::shared_ptr<SimulationState>(ptrToOwn)) {}

  /// @brief Return the data element at the given indices
  std::complex<double> operator[](std::size_t idx);
  std::complex<double> operator()(std::size_t idx, std::size_t jdx);

  /// @brief Dump the state to standard out
  void dump();
  void dump(std::ostream &os);

  /// @brief Return the dimensions of the state vector or density
  /// matrix.
  std::vector<std::size_t> get_shape() const;

  /// @brief Return a constant handle to the data holder.
  const SimulationState *data_holder() const { return internal.get(); }

  /// @brief Compute the overlap of this state
  /// with the other one.
  double overlap(state &other);

  /// @brief Compute the overlap of this state with user-provided host vector
  /// data.
  double overlap(const std::vector<complex128> &hostData);

  /// @brief Compute the overlap of this state with user-provided host vector
  /// data.
  double overlap(const std::vector<complex64> &hostData);

  /// @brief Compute the overlap of this state with a user-provided host or
  /// device data pointer. The size of this data array is assumed to be the same
  /// size as this state's data.
  double overlap(complex128 *deviceOrHostPointer, std::size_t numElements);

  /// @brief Compute the overlap of this state with a user-provided host or
  /// device data pointer. The size of this data array is assumed to be the same
  /// size as this state's data.
  double overlap(complex64 *deviceOrHostPointer, std::size_t numElements);

  ~state();
};

} // namespace cudaq