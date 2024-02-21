/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ quick_start.cpp && ./a.out`
// [Begin Documentation]
#include <cudaq.h>

__qpu__ void kernel() { 
  cudaq::qubit qubit; 
  h(qubit); 
  mz(qubit); 
} 

int main() { 
  auto result = cudaq::sample(kernel); 
  result.dump();  // { 1:500 0:500 }
} 
// [End Documentation]