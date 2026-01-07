/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target qpp-cpu --enable-mlir %s -o %t && CUDAQ_TIMING_TAGS=5 %t | FileCheck %s
// clang-format on

// This test performs per-gate timing measurements. The FileCheck criteria is
// simply "did it run to completion?", but we will do additional timing tests
// elsewhere.

#include "nvqir/CircuitSimulator.h"
#include <cudaq.h>
#include <iostream>

// The number of the qubits in the simulation
#ifndef NUM_QUBITS
#define NUM_QUBITS 10
#endif

// The number of times to execute apply each gate (averaging over the timing
// results)
#ifndef RUN_AVG
#define RUN_AVG 10
#endif

#ifndef DEBUG
// #define DEBUG
#endif

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

// Global data for timing functions
std::vector<std::string> g_tagName{"h", "rx", "cnot"}; // for debug prints
std::vector<double> g_hVec;
std::vector<double> g_rxVec;
std::vector<std::vector<double>> g_cnotMatrix;
#ifdef _LIBCPP_VERSION
std::chrono::high_resolution_clock::time_point g_time;
#else
std::chrono::system_clock::time_point g_time;
#endif

// We use constants to store data in the above global data structures.
#define KERNEL_H 0
#define KERNEL_RX 1
#define KERNEL_CNOT 2

void timer_start(int tag) {
  // Flush before starting the timer to get accurate timing
  nvqir::getCircuitSimulatorInternal()->flushGateQueue();
  g_time = std::chrono::high_resolution_clock::now();
}

void timer_stop(int tag, int count) {
  // Flush before stopping the timer to get accurate timing
  nvqir::getCircuitSimulatorInternal()->flushGateQueue();
  auto tStop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = tStop - g_time;
  double avg_ms = diff.count() * 1000.0 / count;
#ifdef DEBUG
  std::cout << "Duration[" << g_tagName[tag] << "] was " << avg_ms
            << " ms per iteration\n";
#endif
  if (tag == KERNEL_H)
    g_hVec.push_back(avg_ms);
  else if (tag == KERNEL_RX)
    g_rxVec.push_back(avg_ms);
  else
    g_cnotMatrix.back().push_back(avg_ms);
}

void kernel_h(int target) __qpu__ {
  cudaq::qvector q(NUM_QUBITS);
  h(q[0]); // warm-up operation
  timer_start(KERNEL_H);
  for (int j = 0; j < RUN_AVG; j++)
    h(q[target]);
  timer_stop(KERNEL_H, RUN_AVG);
}

void kernel_rx(int target) __qpu__ {
  cudaq::qvector q(NUM_QUBITS);
  h(q[0]); // warm-up operation
  timer_start(KERNEL_RX);
  for (int j = 0; j < RUN_AVG; j++)
    rx(0.1, q[0]);
  timer_stop(KERNEL_RX, RUN_AVG);
}

void kernel_cnot(int control, int target) __qpu__ {
  cudaq::qvector q(NUM_QUBITS);
  h(q[0]); // warm-up operation
  timer_start(KERNEL_CNOT);
  for (int j = 0; j < RUN_AVG; j++)
    cudaq::x<cudaq::ctrl>(q[control], q[target]);

  timer_stop(KERNEL_CNOT, RUN_AVG);
}

int main() {

  g_hVec.reserve(NUM_QUBITS);
  for (int t = 0; t < NUM_QUBITS; t++)
    cudaq::sample(kernel_h, t);

  g_rxVec.reserve(NUM_QUBITS);
  for (int t = 0; t < NUM_QUBITS; t++)
    cudaq::sample(kernel_rx, t);

  g_cnotMatrix.reserve(NUM_QUBITS);
  for (int c = 0; c < NUM_QUBITS; c++) {
    g_cnotMatrix.push_back({});
    g_cnotMatrix.back().reserve(NUM_QUBITS);
    for (int t = 0; t < NUM_QUBITS; t++) {
      if (c == t)
        g_cnotMatrix.back().push_back(0); // N/A
      else
        cudaq::sample(kernel_cnot, c, t);
    }
  }

#ifdef DEBUG
  std::cout << "H timings (one value per target qubit):\n";
  for (auto val : g_hVec)
    std::cout << val << ",";
  std::cout << '\n';
  std::cout << "RX timings (one value per target qubit):\n";
  for (auto val : g_rxVec)
    std::cout << val << ",";
  std::cout << '\n';
  std::cout << "CNOT timings (one value per [control,target] pair):\n";
  for (auto &vecConstControl : g_cnotMatrix) {
    for (auto val : vecConstControl)
      std::cout << val << ",";
    std::cout << '\n';
  }
#endif

  // Average a vector, excluding any 0's from the average
  auto avgVector = [](const std::vector<double> &vec) {
    double sum = 0.0;
    int count = 0;
    for (auto val : vec) {
      if (val > 0) {
        sum += val;
        count++;
      }
    }
    return count > 0 ? sum / count : 0.0;
  };
  // Average a matrix, excluding any 0's from the average
  auto avgMatrix = [&](const std::vector<std::vector<double>> &mat) {
    double sum = 0.0;
    int count = 0;
    for (auto &row : mat) {
      if (auto rowAvg = avgVector(row)) {
        sum += rowAvg;
        count++;
      }
    }
    return count > 0 ? sum / count : 0.0;
  };

  auto avgH = avgVector(g_hVec);
  auto avgRx = avgVector(g_rxVec);
  auto avgCNOT = avgMatrix(g_cnotMatrix);
  std::cout << "avgH = " << avgH << " ms\n";
  std::cout << "avgRx = " << avgRx << " ms\n";
  std::cout << "avgCNOT = " << avgCNOT << " ms\n";

  return 0;
}

// CHECK: avgH = {{.*}} ms
// CHECK: avgRx = {{.*}} ms
// CHECK: avgCNOT = {{.*}} ms
// CHECK: {{.*}} CircuitSimulator 'qpp' Total Program Metrics [tag=5]:
// CHECK: {{.*}} Gate Count = 1210
// CHECK: {{.*}} Control Count = 900
// CHECK: {{.*}} Target Count = 1210
// CHECK: {{.*}} State Vector I/O (GB) = 0.024904
// CHECK: {{.*}} State Vector GFLOPs = 0.010895
