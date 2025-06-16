/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ mz_qumode_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qvector<3> qumodes(2);
  // [Begin MZ Op]
  mz(qumodes);

  return 0;
}
// [End MZ Op]

