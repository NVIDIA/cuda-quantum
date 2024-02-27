/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <unordered_map>

#include "cutensornet.h"
#include "tensornet_state.h"
#include "tensornet_utils.h"
#include "timing_utils.h"

#include "common/SimulationState.h"

namespace nvqir {

class MPSSimulationState : public TensorNetSimulationState {

public:

  MPSSimulationState(TensorNetState *inState,
                     const std::vector<MPSTensor> & mpsTensors);

  MPSSimulationState(const MPSSimulationState &) = delete;
  MPSSimulationState & operator=(const MPSSimulationState &) = delete;
  MPSSimulationState(MPSSimulationState &&) noexcept = default;
  MPSSimulationState & operator=(MPSSimulationState && ) noexcept = default;

  virtual ~MPSSimulationState();

  double overlap(const cudaq::SimulationState &other) override;

protected:

  void deallocate();

  std::vector<MPSTensor> m_mpsTensors;
};

} // namespace nvqir