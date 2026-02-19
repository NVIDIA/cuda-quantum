/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

__qpu__ int random_bit_add(int a, int b) {
  cudaq::qubit q;
  // Hadamard to create superposition
  h(q);
  int result = cudaq::device_call<int>(/*device_id*/ 0, "add_op", a, b);
  return mz(q) + result;
}

__qpu__ int random_bit_minus(int a, int b) {
  cudaq::qubit q;
  // Hadamard to create superposition
  h(q);
  int result = cudaq::device_call<int>(/*device_id*/ 0, "sub_op", a, b);
  return mz(q) + result;
}

__qpu__ int random_bit_multiply(int a, int b) {
  cudaq::qubit q;
  // Hadamard to create superposition
  h(q);
  int result = cudaq::device_call<int>(/*device_id*/ 0, "mul_op", a, b);
  return mz(q) + result;
}

__qpu__ int random_bit_invalid(int a, int b) {
  cudaq::qubit q;
  // Hadamard to create superposition
  h(q);
  // This is an unknown operation that the server won't recognize, which should
  // trigger error handling.
  int result = cudaq::device_call<int>(/*device_id*/ 0, "invalid_op", a, b);
  return mz(q) + result;
}

int main() {
  const auto checkExpectedResults = [](const std::vector<int> &results,
                                       int val) {
    // Expected results are val and val+1 (50-50), since the qubit measurement
    // can add 0 or 1 to the result.
    const auto countVal = std::count(results.begin(), results.end(), val);
    const auto countValPlusOne =
        std::count(results.begin(), results.end(), val + 1);
    const auto total = results.size();
    const auto ratioVal = static_cast<double>(countVal) / total;
    const auto ratioValPlusOne = static_cast<double>(countValPlusOne) / total;

    return (ratioVal > 0.25 && ratioVal < 0.75) &&
           (ratioValPlusOne > 0.25 && ratioValPlusOne < 0.75) &&
           (countVal + countValPlusOne == total);
  };

  // Test some random inputs of a and b
  for (int a : {3, 5, 7}) {
    for (int b : {2, 4, 6}) {
      auto results = cudaq::run(100, random_bit_add, a, b);
      if (!checkExpectedResults(results, a + b)) {
        std::cerr << "Random bit add results failed for inputs " << a << " and "
                  << b << ". Results: ";
        for (const auto &res : results) {
          std::cerr << res << " ";
        }
        std::cerr << std::endl;
        return 1;
      }
      std::cout << "Random bit add passed for inputs " << a << " and " << b
                << ". \n";

      results = cudaq::run(100, random_bit_minus, a, b);
      if (!checkExpectedResults(results, a - b)) {
        std::cerr << "Random bit minus results failed for inputs " << a
                  << " and " << b << ". Results: ";
        for (const auto &res : results) {
          std::cerr << res << " ";
        }
        std::cerr << std::endl;
        return 1;
      }
      std::cout << "Random bit minus passed for inputs " << a << " and " << b
                << ". \n";

      results = cudaq::run(100, random_bit_multiply, a, b);
      if (!checkExpectedResults(results, a * b)) {
        std::cerr << "Random bit multiply results failed for inputs " << a
                  << " and " << b << ". Results: ";
        for (const auto &res : results) {
          std::cerr << res << " ";
        }
        std::cerr << std::endl;
        return 1;
      }
      std::cout << "Random bit multiply passed for inputs " << a << " and " << b
                << ". \n";
    }
  }

  try {
    auto results = cudaq::run(100, random_bit_invalid, 1, 2);
    std::cerr
        << "Expected random_bit_invalid to throw an exception, but it did not."
        << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::string expectedMessagePart =
        "Function invalid_op not found in JIT engine.";
    if (std::string(e.what()).find(expectedMessagePart) == std::string::npos) {
      std::cerr << "random_bit_invalid threw an exception, but the message was "
                   "unexpected: "
                << e.what() << std::endl;
      return 1;
    }
    // If we reach here, the exception message was as expected, so we can
    // consider this a pass.
    std::cout << "random_bit_invalid correctly threw an exception for unknown "
                 "operation. Exception message: "
              << e.what() << std::endl;
  }
  return 0;
}
