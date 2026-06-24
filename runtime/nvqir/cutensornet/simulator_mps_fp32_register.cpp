/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define TENSORNET_FP32

#include "simulator_mps.h"

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS<float>, tensornet_mps_fp32)
