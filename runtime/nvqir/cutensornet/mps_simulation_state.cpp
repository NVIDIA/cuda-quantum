/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"

namespace nvqir {

MPSSimulationState::MPSSimulationState(TensorNetState *inState,
                                       int64_t inMaxBond, double inAbsCutoff,
                                       double inRelCutoff)
    // FIXME flip this to false to use MPS-only overlap
    : TensorNetSimulationState(inState,
                               /*auto-gen state vector stop-gap*/ true),
      maxBond(inMaxBond), absCutoff(inAbsCutoff), relCutoff(inRelCutoff) {}

double MPSSimulationState::overlap(const cudaq::SimulationState &other) {

  if (other.getDataShape() != getDataShape())
    throw std::runtime_error("[tensornet-state] overlap error - other state "
                             "dimension not equal to this state dimension.");

  if (other.isDeviceData())
    throw std::runtime_error("[tensornet-state] cannot compute "
                             "overlap with GPU state data yet.");

  return std::abs(
      Eigen::Map<Eigen::VectorXcd>(
          const_cast<cudaq::complex128 *>(stateData.data()), stateData.size())
          .transpose()
          .dot(Eigen::Map<Eigen::VectorXcd>(
              reinterpret_cast<cudaq::complex128 *>(other.ptr()),
              stateData.size()))
          .real());

  /* FIXME Comment out the above and uncomment the code below

  // Cast the incoming state to an MPS simulation state.
  const auto &otherMpsState = dynamic_cast<const MPSSimulationState &>(other);

  // Get our MPS tensors
  auto ourTensors = state->factorizeMPS(maxBond, absCutoff, relCutoff);

  // Get the input state's MPS tensors
  auto otherTensors =
      otherMpsState.state->factorizeMPS(maxBond, absCutoff, relCutoff);

  // FIXME COMPUTE OVERLAP
  // You also have state->m_cutnHandle, state->m_quantumState (cutensornetState_t)

  // Cleanup the MPS tensor data
  for (auto &tensor : ourTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  ourTensors.clear();

  for (auto &tensor : otherTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  otherTensors.clear();

  return 0.0;
  */
}
} // namespace nvqir