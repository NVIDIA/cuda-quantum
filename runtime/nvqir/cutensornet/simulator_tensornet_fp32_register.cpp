/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#define TENSORNET_FP32

// GCC 12 emits a spurious -Wstringop-overflow false positive inside
// std::copy<size_t*> inlined from SimulatorTensorNetBase::swap().
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include "simulator_tensornet.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif

/// Register this Simulator class with NVQIR under name "tensornet-fp32"
extern "C" {
nvqir::CircuitSimulator *getCircuitSimulator_tensornet_fp32() {
  thread_local static auto simulator =
      std::make_unique<nvqir::SimulatorTensorNet<float>>();
  // Handle multiple runtime __nvqir__setCircuitSimulator calls before/after MPI
  // initialization. If the static simulator instance was created before MPI
  // initialization, it needs to be reset to support MPI if needed.
  if (cudaq::mpi::is_initialized() && !simulator->m_cutnMpiInitialized) {
    // Reset the static instance to pick up MPI.
    simulator.reset(new nvqir::SimulatorTensorNet<float>());
  }
  return simulator.get();
}
nvqir::CircuitSimulator *getCircuitSimulator() {
  return getCircuitSimulator_tensornet_fp32();
}
}
