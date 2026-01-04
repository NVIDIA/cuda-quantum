/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/RuntimeTarget.h"
#include "cudaq/platform/qpu.h"
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

using namespace cudaq;

class DummyQPU : public cudaq::QPU {
public:
  DummyQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {}

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    return {};
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {}

  void resetExecutionContext() override {}
};

class DummyMQPUPlatform : public cudaq::quantum_platform {
public:
  DummyMQPUPlatform(std::size_t numQPUs) {
    for (std::size_t i = 0; i < numQPUs; ++i) {
      platformQPUs.emplace_back(std::make_unique<DummyQPU>());
    }
  }
};

/// Test that execution contexts are thread-local.
/// Each thread sets its own context, verifies it can retrieve it,
/// and then resets it.
TEST(ExecutionContextThreadTester, checkThreadLocalContext) {
  constexpr int numThreads = 4;
  DummyMQPUPlatform platform(numThreads);

  std::atomic<int> successCount{0};
  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&platform, &successCount, i]() {
      // Create a unique execution context for this thread.
      // Use qpuId to give each context a unique identifier.
      ExecutionContext ctx("sample", 1, /*qpuId=*/i);
      ctx.batchIteration = static_cast<std::size_t>(i);

      // Set the context on the platform for this thread.
      platform.set_exec_ctx(&ctx);

      // Retrieve the context and verify it's the one we set.
      ExecutionContext *retrieved = platform.get_exec_ctx();
      if (retrieved == &ctx &&
          retrieved->batchIteration == static_cast<std::size_t>(i)) {
        successCount.fetch_add(1, std::memory_order_relaxed);
      }

      // Reset the execution context.
      platform.reset_exec_ctx();

      // After reset, context should be null.
      if (platform.get_exec_ctx() == nullptr) {
        successCount.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Wait for all threads to complete.
  for (auto &t : threads) {
    t.join();
  }

  // Each thread should have 2 successful checks:
  // 1. Context was correctly set and retrieved.
  // 2. Context was null after reset.
  EXPECT_EQ(successCount.load(), numThreads * 2);
}
