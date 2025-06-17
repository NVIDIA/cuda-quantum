/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ beam_splitter_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qvector<3> q(2);
  // [Begin Beam Splitter Op]
  beam_splitter(q[0], q[1], 0.34);

  return 0;
}
// [End Beam Splitter Op]

