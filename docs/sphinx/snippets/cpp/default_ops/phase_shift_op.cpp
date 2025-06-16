/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ phase_shift_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qvector<4> q(1);
  phase_shift(q[0], 0.17);

  return 0;
}

