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

// CTest checks process exit to catch failures after main returns.
CUDAQ_TEST(TensorNetShutdownTester, SeededWorkerExitsCleanly) {
  // Create the queue before CUDA so it outlives CUDA unless stopped early.
  static cudaq::QuantumExecutionQueue queue;
  std::promise<void> ready;
  auto done = ready.get_future();
  cudaq::QuantumTask task = [&] {
    try {
      cudaq::set_random_seed(123);
      ready.set_value();
    } catch (...) {
      ready.set_exception(std::current_exception());
    }
  };
  queue.enqueue(task);
  EXPECT_NO_THROW(done.get());
}

CUDAQ_TEST(TensorNetShutdownTester, AsyncObserveExitsCleanly) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
  };
  auto result = cudaq::observe_async(kernel, cudaq::spin_op::z(0)).get();
  EXPECT_NEAR(result.expectation(), -1.0, 1e-6);
}

CUDAQ_TEST(TensorNetShutdownTester, AsyncSampleExitsCleanly) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  };
  auto result = cudaq::sample_async(kernel).get();
  EXPECT_EQ(result.size(), 1u);
  EXPECT_EQ(result.count("1"), 1000u);
}

CUDAQ_TEST(TensorNetShutdownTester, AsyncStateExitsCleanly) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
  };
  auto state = cudaq::get_state_async(kernel).get();
  EXPECT_EQ(state.get_num_qubits(), 1u);
  EXPECT_NEAR(std::abs(state[1]), 1.0, 1e-6);
}
