/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ annihilate_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qvector<3> q(1);
  annihilate(q[0]);

  return 0;
}

