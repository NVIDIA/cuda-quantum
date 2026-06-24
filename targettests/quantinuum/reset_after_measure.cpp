/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>
#include <iostream>

void explicit_reset_after_mz() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
  x(q);
}

void auto_reset_injection() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  x(q);
}

void reuse1() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  auto res = mz(q);
  if (res[0]) {
    h(q);
  }
}

void foo(cudaq::qview<> q) __qpu__ { t(q); }

void reuse2() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
  foo(q);
}

// Use q[1] only
void bar(cudaq::qview<> q) __qpu__ { t(q[1]); }

void reuse3() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
  bar(q);
}

// =============================================
// Qubit aliasing tests
// =============================================
void reuse4() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
  auto alias = cudaq::qview<>(q);
  t(alias);
}

void reuse5() __qpu__ {
  cudaq::qvector q(2);
  auto &q1 = q.front();
  auto &q2 = q.back();
  h(q1);
  cx(q1, q2);
  mz(q);
  t(q);
}

void reuse6() __qpu__ {
  cudaq::qvector q(4);
  auto &q1 = q.front();
  auto rest = q.slice(1, 3);
  h(q1);
  for (int i = 0; i < rest.size(); i++)
    cx(q1, rest[i]);
  mz(q);
  t(q);
}

void no_reuse7(int start, int size) __qpu__ {
  cudaq::qvector q(4);
  auto &q1 = q.front();
  // Variable slice
  auto rest = q.slice(start, size);
  h(q1);
  for (int i = 0; i < rest.size(); i++)
    cx(q1, rest[i]);
  mz(q);
}

void reuse7(int start, int size) __qpu__ {
  cudaq::qvector q(4);
  auto &q1 = q.front();
  // Variable slice
  auto rest = q.slice(start, size);
  h(q1);
  for (int i = 0; i < rest.size(); i++)
    cx(q1, rest[i]);
  mz(q);
  t(rest);
}

int main() {
  {
    auto gateCounts = cudaq::estimate_resources(explicit_reset_after_mz);
    if (gateCounts.count("reset") == 1)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCountsTrue =
        cudaq::estimate_resources([]() { return true; }, auto_reset_injection);

    // One reset is added automatically before the x gate. There are 2 X gates
    // as the conditional X after the reset is taken.
    if (gateCountsTrue.count("reset") == 1 && gateCountsTrue.count("x") == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  {
    // The false path does not take the conditional X after the reset, hence
    // only one X gate is counted.
    auto gateCountsFalse =
        cudaq::estimate_resources([]() { return false; }, auto_reset_injection);
    if (gateCountsFalse.count("reset") == 1 && gateCountsFalse.count("x") == 1)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCountsTrue =
        cudaq::estimate_resources([]() { return true; }, reuse1);
    gateCountsTrue.dump();
    if (gateCountsTrue.count("reset") == 2 &&
        gateCountsTrue.count_controls("x", 0) == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  {
    auto gateCountsFalse =
        cudaq::estimate_resources([]() { return false; }, reuse1);
    gateCountsFalse.dump();
    if (gateCountsFalse.count("reset") == 2 &&
        gateCountsFalse.count_controls("x", 0) == 0)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCountsTrue =
        cudaq::estimate_resources([]() { return true; }, reuse2);
    gateCountsTrue.dump();
    if (gateCountsTrue.count("reset") == 2 &&
        gateCountsTrue.count_controls("x", 0) == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  {
    auto gateCountsFalse =
        cudaq::estimate_resources([]() { return false; }, reuse2);
    gateCountsFalse.dump();
    if (gateCountsFalse.count("reset") == 2 &&
        gateCountsFalse.count_controls("x", 0) == 0)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCountsTrue =
        cudaq::estimate_resources([]() { return true; }, reuse3);
    gateCountsTrue.dump();
    if (gateCountsTrue.count("reset") == 1 &&
        gateCountsTrue.count_controls("x", 0) == 1)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  {
    auto gateCountsFalse =
        cudaq::estimate_resources([]() { return false; }, reuse3);
    gateCountsFalse.dump();
    if (gateCountsFalse.count("reset") == 1 &&
        gateCountsFalse.count_controls("x", 0) == 0)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCounts = cudaq::estimate_resources(reuse4);
    gateCounts.dump();
    // Two resets
    if (gateCounts.count("reset") == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCounts = cudaq::estimate_resources(reuse5);
    gateCounts.dump();
    // Two resets
    if (gateCounts.count("reset") == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCounts = cudaq::estimate_resources(reuse6);
    gateCounts.dump();
    // 4 resets
    if (gateCounts.count("reset") == 4)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }

  {
    auto gateCounts = cudaq::estimate_resources(no_reuse7, 1, 2);
    gateCounts.dump();

    if (gateCounts.count("reset") == 0)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  {
    auto gateCounts = cudaq::estimate_resources(reuse7, 1, 2);
    gateCounts.dump();
    // 2 resets (sliced qubits)
    if (gateCounts.count("reset") == 2)
      std::cout << "success\n";
    else
      std::cout << "failure\n";
    // CHECK: success
  }
  return 0;
}
