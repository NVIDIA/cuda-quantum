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
                                       const std::vector<MPSTensor> & mpsTensors) :
  // FIXME flip this to false to use MPS-only overlap
  TensorNetSimulationState(inState,
                           /*auto-gen state vector stop-gap*/ true),
  m_mpsTensors(mpsTensors)
{
}

MPSSimulationState::~MPSSimulationState()
{
  deallocate();
}

double MPSSimulationState::overlap(const cudaq::SimulationState &other) {

  if (other.getDataShape() != getDataShape())
    throw std::runtime_error("[tensornet-state] overlap error - other state "
                             "dimension is not equal to this state dimension.");

  // Cast the incoming state to an MPS simulation state
  //const auto &mpsOther = dynamic_cast<const MPSSimulationState &>(other);

  // FIXME COMPUTE OVERLAP
  // You also have state->m_cutnHandle, state->m_quantumState (cutensornetState_t)

  return 0.0;
}

void MPSSimulationState::deallocate()
{
  for (auto &tensor : m_mpsTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  m_mpsTensors.clear();
}

} // namespace nvqir