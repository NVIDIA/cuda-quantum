/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <iostream>

__qpu__ int loop_payload_stress() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  for (int pass = 0; pass < 2; ++pass)
    for (int i = 0; i < width; ++i)
      x(q[i]);

  int i = 0;
  while (i < width) {
    x(q[i]);
    ++i;
  }

  int result = 0;
  if (mz(q[0]))
    result |= 1;
  if (mz(q[1]))
    result |= 2;
  if (mz(q[2]))
    result |= 4;
  if (mz(q[3]))
    result |= 8;

  return result;
}

int main() {
  const auto results = cudaq::run(3, loop_payload_stress);
  for (auto result : results) {
    if (result != 15) {
      std::cerr << "loop_payload_stress expected 15, got " << result << "\n";
      return 1;
    }
  }

  std::cout << "Loop payload stress passed.\n";
  return 0;
}
