/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <string_view>

__qpu__ bool synthesized_early_return_leaf(bool condition) {
  cudaq::qubit q;
  h(q);
  if (condition)
    return true;
  x(q);
  return false;
}

__qpu__ bool synthesized_early_return_middle(bool condition) {
  return synthesized_early_return_leaf(condition);
}

__qpu__ bool synthesized_early_return(bool condition) {
  return synthesized_early_return_middle(condition);
}

__qpu__ bool synthesized_break(bool condition) {
  cudaq::qubit q;
  for (int i = 0; i < 2; ++i) {
    h(q);
    if (condition)
      break;
    x(q);
  }
  return false;
}

__qpu__ bool synthesized_continue(bool condition) {
  cudaq::qubit q;
  for (int i = 0; i < 2; ++i) {
    h(q);
    if (condition)
      continue;
    x(q);
  }
  return false;
}

__qpu__ bool measurement_early_return() {
  cudaq::qubit q;
  x(q);
  if (mz(q))
    return true;
  h(q);
  return false;
}

__qpu__ bool measurement_break() {
  cudaq::qubit q;
  for (int i = 0; i < 2; ++i) {
    x(q);
    if (mz(q))
      break;
    h(q);
  }
  return false;
}

__qpu__ bool measurement_continue() {
  cudaq::qubit q;
  for (int i = 0; i < 2; ++i) {
    x(q);
    if (mz(q))
      continue;
    h(q);
  }
  return false;
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;

  const std::string_view testCase = argv[1];
  if (testCase == "early-return")
    cudaq::run(1, synthesized_early_return, true);
  else if (testCase == "break")
    cudaq::run(1, synthesized_break, true);
  else if (testCase == "continue")
    cudaq::run(1, synthesized_continue, true);
  else if (testCase == "measurement-return")
    cudaq::run(1, measurement_early_return);
  else if (testCase == "measurement-break")
    cudaq::run(1, measurement_break);
  else if (testCase == "measurement-continue")
    cudaq::run(1, measurement_continue);
  else
    return 2;
}
