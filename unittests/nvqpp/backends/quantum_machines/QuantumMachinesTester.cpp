/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ServerHelper.h"
#include "math.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithm.h"
#include "llvm/Support/Base64.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <stdlib.h>
#include <string_view>
#include <vector>

CUDAQ_TEST(QuantumMachinesTester, minimal3Hadamard) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(3);
  kernel.h(qubit[0]);
  kernel.h(qubit[1]);
  kernel.h(qubit[2]);

  auto counts = cudaq::sample(1000, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

CUDAQ_TEST(QuantumMachinesTester, resetAndH) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(3);
  for (int i = 0; i < 3; i++) {
    kernel.reset(qubit[0]);
  }
  for (int i = 0; i < 3; i++) {
    kernel.h(qubit[1]);
  }

  auto counts = cudaq::sample(1000, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

CUDAQ_TEST(QuantumMachinesTester, gates) {
  auto kernel = cudaq::make_kernel();

  int qubit_count = 5;
  auto qvector = kernel.qalloc(qubit_count);
  for (int i = 0; i < qubit_count; i++) {
    kernel.reset(qvector[i]);
  }
  kernel.t(qvector[0]);
  kernel.s(qvector[1]);
  kernel.r1(1.1853982, qvector[2]);
  kernel.x(qvector[3]);
  kernel.y(qvector[4]);

  auto counts = cudaq::sample(1001, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

namespace {
constexpr std::string_view currentQuakeIR = R"mlir(module {
  func.func @kernel(%arg0: !cc.sequence<i64>) {
    quake.log_output %arg0 : (!cc.sequence<i64>) -> ()
    return
  }
})mlir";

std::string createSubmittedIR(cudaq::ServerHelper &helper) {
  std::vector<cudaq::KernelExecution> executions{
      {.name = "kernel", .code = llvm::encodeBase64(currentQuakeIR)}};
  const auto payload = helper.createJob(executions);
  const auto &jobs = std::get<2>(payload);
  if (jobs.size() != 1)
    throw std::runtime_error("Expected one QM job payload.");

  std::vector<char> decodedContent;
  const auto encodedContent = jobs.front().at("content").get<std::string>();
  if (auto error = llvm::decodeBase64(encodedContent, decodedContent))
    throw std::runtime_error("Failed to decode QM test payload.");
  return {decodedContent.begin(), decodedContent.end()};
}
} // namespace

CUDAQ_TEST(QuantumMachinesTester, rewritesLegacySpellings) {
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("quantum_machines");
  ASSERT_TRUE(helper);
  helper->initialize({{"url", "http://localhost:62448"}});

  const auto submittedIR = createSubmittedIR(*helper);
  EXPECT_NE(submittedIR.find("!cc.stdvec<i64>"), std::string::npos);
  EXPECT_NE(submittedIR.find("cc.log_output %arg0 : !cc.stdvec<i64>"),
            std::string::npos);
  EXPECT_EQ(submittedIR.find("cc.sequence"), std::string::npos);
  EXPECT_EQ(submittedIR.find("quake.log_output"), std::string::npos);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
