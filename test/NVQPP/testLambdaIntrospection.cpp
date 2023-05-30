/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o out_testlambdaI.x && ./out_testlambdaI.x | FileCheck %s

#include <cudaq.h>

struct ghz {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
  }
};

void function() {
  // Can have ll defined more than once
  // but in non-inclusive scopes
  auto lambdaFunctionParent = []() __qpu__ {
    cudaq::qubit q;
    x(q);
  };
  printf("5: %s", cudaq::get_quake(lambdaFunctionParent).data());
}

int main() {

  auto lambdaMain = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    h(q);
    mz(q);
  };

  auto lambdaAgain = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    h(q);
    mz(q);
  };

  lambdaMain();

  printf("2: %s", cudaq::get_quake(lambdaMain).data());
  printf("3: %s", cudaq::get_quake(lambdaAgain).data());
  printf("4: %s", cudaq::get_quake(ghz{}).data());

  function();

  return 0;
}

// CHECK: 2: module { func.func @__nvqpp__mlirgen__Z4mainE3$_0() attributes {
// CHECK: 3: module { func.func @__nvqpp__mlirgen__Z4mainE3$_1() attributes {
// CHECK: 4: module { func.func @__nvqpp__mlirgen__ghz{{.*}}() attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
// CHECK: 5: module { func.func @__nvqpp__mlirgen__Z8functionvE3$_0() attributes {
