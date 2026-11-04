/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include <future>

namespace nvqir {
class CircuitSimulator;
}
extern "C" nvqir::CircuitSimulator *CONCAT(getCircuitSimulator_,
                                           NVQIR_BACKEND_NAME)();

CUDAQ_TEST(CommunicatorLifetimeTester, WorkerSurvivesUntilProcessExit) {
  // Create the queue first so it outlives MPI and the provider unless stopped.
  // CTest checks process exit to catch crashes after RUN_ALL_TESTS() returns.
  static cudaq::QuantumExecutionQueue queue;
  std::promise<bool> ready;
  auto done = ready.get_future();
  cudaq::QuantumTask task = [&] {
    try {
      // Keep the simulator alive until the worker exits.
      auto *simulator = CONCAT(getCircuitSimulator_, NVQIR_BACKEND_NAME)();
      ready.set_value(simulator && cudaq::mpi::is_initialized());
    } catch (...) {
      ready.set_exception(std::current_exception());
    }
  };
  queue.enqueue(task);
  EXPECT_TRUE(done.get());
}
