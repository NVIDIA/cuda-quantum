/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cudaq.h>

int main() {
  auto kernel = cudaq::make_kernel();
  auto q = kernel.qalloc();
  kernel.h(q);
  kernel.mz(q);
  auto counts = cudaq::sample(kernel);
  return counts.count("0") + counts.count("1") > 0 ? 0 : 1;
}
