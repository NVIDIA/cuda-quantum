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

protected:
  int64_t maxBond = 0;
  double absCutoff = 1e-6;
  double relCutoff = 1e-6;

public:
  MPSSimulationState(TensorNetState *inState, int64_t inMaxBond,
                     double inAbsCutoff, double inRelCutoff);

  double overlap(const cudaq::SimulationState &other) override;
};

} // namespace nvqir