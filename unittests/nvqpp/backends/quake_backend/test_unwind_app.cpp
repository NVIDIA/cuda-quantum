/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <vector>

__qpu__ int foo(bool returnEarly) {
  int i = 0;

  for (; i < 4; ++i) {
    if (returnEarly)
      return 1;
  }

  return i;
}

__qpu__ int branchReturns(bool firstBranch) {
  if (firstBranch)
    return 7;
  else
    return 9;
}

__qpu__ int loopControl(int skip, int stop) {
  int i = 0;
  int total = 0;

  while (i < 6) {
    if (i == stop)
      break;
    if (i == skip) {
      ++i;
      continue;
    }
    total += i;
    ++i;
  }

  return total;
}

bool hasExpectedResult(const std::vector<int> &results, int expected) {
  return results.size() == 1 && results.front() == expected;
}

int main() {
  const auto earlyResults = cudaq::run(1, foo, true);
  if (!hasExpectedResult(earlyResults, 1))
    return 1;

  const auto loopResults = cudaq::run(1, foo, false);
  if (!hasExpectedResult(loopResults, 4))
    return 2;

  const auto firstBranchResults = cudaq::run(1, branchReturns, true);
  if (!hasExpectedResult(firstBranchResults, 7))
    return 3;

  const auto secondBranchResults = cudaq::run(1, branchReturns, false);
  if (!hasExpectedResult(secondBranchResults, 9))
    return 4;

  const auto loopControlResults = cudaq::run(1, loopControl, 2, 5);
  if (!hasExpectedResult(loopControlResults, 8))
    return 5;

  const auto completeLoopResults = cudaq::run(1, loopControl, 9, 9);
  return hasExpectedResult(completeLoopResults, 15) ? 0 : 6;
}
