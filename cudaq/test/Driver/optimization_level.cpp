/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -c %s -v -O2 -o %t.o | FileCheck %s --check-prefix=HAS-LEVEL
// RUN: nvq++ -c %s -v -o %t.o | FileCheck %s --check-prefix=NO-LEVEL
// RUN: nvq++ %s -O3 -o %t && %t | FileCheck %s --check-prefix=O3
// RUN: nvq++ %s -O0 -o %t && %t | FileCheck %s --check-prefix=O0
// RUN: nvq++ %s -o %t && %t | FileCheck %s --check-prefix=DEFAULT

// HAS-LEVEL: cudaq-quake {{.*}}-D NVQPP_OPTIMIZATION_LEVEL=2
// NO-LEVEL-NOT: NVQPP_OPTIMIZATION_LEVEL

// O3: optimization level: 3
// O0: optimization level: 0
// DEFAULT: optimization level: 1

#include "common/CompileOptions.h"
#include "cudaq.h"
#include <iostream>

int main() {
  cudaq::CompileOptions options;
  options.populateDefaults();
  std::cout << "optimization level: "
            << static_cast<unsigned>(*options.optimizationLevel) << "\n";
  return 0;
}
