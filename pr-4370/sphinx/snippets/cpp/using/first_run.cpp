/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//[Begin Run1]
#include <algorithm>
#include <cudaq.h>
#include <iostream>
#include <map>
#include <numeric>

// Define a quantum kernel that returns an integer
__qpu__ int ghz_kernel(int qubit_count) {
  // Allocate qubits
  cudaq::qvector qubits(qubit_count);

  // Create GHZ state
  h(qubits[0]);
  for (int i = 1; i < qubit_count; ++i) {
    x<cudaq::ctrl>(qubits[0], qubits[i]);
  }

  // Measure and count the number of qubits in state |1⟩
  int result = 0;
  for (int i = 0; i < qubit_count; ++i) {
    if (mz(qubits[i])) {
      result += 1;
    }
  }

  return result;
}

int main() {
  // Execute the kernel multiple times and collect individual results
  int qubit_count = 3;
  auto results = cudaq::run(10, ghz_kernel, qubit_count);

  std::cout << "Executed " << results.size() << " shots\n";
  std::cout << "Results: ";
  for (auto result : results) {
    std::cout << result << " ";
  }
  std::cout << "\n";
  //[End Run1]

  //[Begin Run2]
  // Count occurrences of each result
  std::map<int, int> value_counts;
  for (auto value : results) {
    value_counts[value]++;
  }

  std::cout << "\nCounts of each result:\n";
  for (auto &[value, count] : value_counts) {
    std::cout << "Result " << value << ": " << count << " times\n";
  }

  // Analyze patterns in the results
  int zero_count = std::count(results.begin(), results.end(), 0);
  int full_count = std::count(results.begin(), results.end(), qubit_count);
  int other_count = results.size() - zero_count - full_count;

  std::cout << "\nGHZ state analysis:\n";
  std::cout << "  All qubits in |0⟩: " << zero_count << " times ("
            << (float)zero_count / results.size() * 100.0 << "%)\n";
  std::cout << "  All qubits in |1⟩: " << full_count << " times ("
            << (float)full_count / results.size() * 100.0 << "%)\n";
  std::cout << "  Other states: " << other_count << " times ("
            << (float)other_count / results.size() * 100.0 << "%)\n";
  //[End Run2]

  //[Begin RunAsync]
  // Define a simple kernel for async execution
  auto simple_kernel = [](float theta) __qpu__ -> bool {
    cudaq::qubit q;
    rx(theta, q);
    return mz(q);
  };

  // Check if we have multiple QPUs available
  // Note: In C++ API, we would check this differently
  // Here we'll use the target setting directly
  bool has_multiple_qpus = false;

  if (has_multiple_qpus) {
    // Set the target to include multiple virtual QPUs
    // In a real application, this would involve proper target configuration

    // Run kernels asynchronously with different parameters
    auto future1 = cudaq::run_async(0, 100, simple_kernel, 0.0);
    auto future2 = cudaq::run_async(1, 100, simple_kernel, 3.14159);

    // Get results when ready
    auto results1 = future1.get();
    auto results2 = future2.get();

    // Analyze the results
    std::cout << "\nAsynchronous execution results:\n";
    int true_count1 = std::count(results1.begin(), results1.end(), true);
    int true_count2 = std::count(results2.begin(), results2.end(), true);

    std::cout << "Kernel with theta=0.0: " << true_count1
              << "/100 times measured |1⟩\n";
    std::cout << "Kernel with theta=π: " << true_count2
              << "/100 times measured |1⟩\n";
  } else {
    // Schedule for execution on the same QPU
    auto future1 = cudaq::run_async(0, 100, simple_kernel, 0.0);
    auto future2 = cudaq::run_async(0, 100, simple_kernel, 3.14159);

    // Get results when ready
    auto results1 = future1.get();
    auto results2 = future2.get();

    // Analyze the results
    std::cout << "\nAsynchronous execution results:\n";
    int true_count1 = std::count(results1.begin(), results1.end(), true);
    int true_count2 = std::count(results2.begin(), results2.end(), true);

    std::cout << "Kernel with theta=0.0: " << true_count1
              << "/100 times measured |1⟩\n";
    std::cout << "Kernel with theta=π: " << true_count2
              << "/100 times measured |1⟩\n";
  }
  //[End RunAsync]

  return 0;
}
