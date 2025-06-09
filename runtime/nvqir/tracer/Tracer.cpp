/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/CircuitSimulator.h"
#include "Tracer.h"

#ifndef __NVQIR_TRACER_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::Tracer, tracer)
#endif

extern "C" {
nvqir::CircuitSimulator *__nvqir__getTracerCircuitSimulator() {
    return new nvqir::Tracer();
}
}