/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Docs]
#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(MyCNOT, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0});

CUDAQ_REGISTER_OPERATION(
    MyXY, 2, 0,
    {0, 0, 0, {0, -1}, 0, 0, {0, 1}, 0, 0, {0, -1}, 0, 0, {0, 1}, 0, 0, 0});

__qpu__ void bell_pair() {
  cudaq::qubit q, r;
  h(q);
  MyCNOT(q, r); // MyCNOT(control, target)
}

__qpu__ void custom_xy_test() {
  cudaq::qubit q, r;
  MyXY(q, r);
  y(r); // undo the prior Y gate on qubit 1
}

int main() {
  auto counts = cudaq::sample(bell_pair);
  counts.dump(); // prints { 11:500 00:500 } (exact numbers will be random)

  counts = cudaq::sample(custom_xy_test);
  counts.dump(); // prints { 10:1000 }
}
// [End Docs]
