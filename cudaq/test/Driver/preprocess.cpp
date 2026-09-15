/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify nvq++ -E (preprocess-only) delegates to the host preprocessor on both
// compile paths: the default target (cudaq-quake) and orca-photonics library
// mode.

// RUN: rm -rf %t && mkdir -p %t

// Default target: -E to stdout.
// RUN: nvq++ -E %s -I%S/Inputs | FileCheck %s --check-prefix=STDOUT

// Default target: -E with -o.
// RUN: nvq++ -E %s -o %t/out.ii -I%S/Inputs
// RUN: [ -e %t/out.ii ]
// RUN: cat %t/out.ii | FileCheck %s --check-prefix=FILE

// Orca-photonics (library mode): -E with -o.
// RUN: nvq++ --target orca-photonics -E %s -o %t/orca.ii -I%S/Inputs
// RUN: [ -e %t/orca.ii ]
// RUN: cat %t/orca.ii | FileCheck %s --check-prefix=ORCA

#include "dep_header.h"
#include <cudaq.h>

int plain_old_function() { return cudaq_test_dep_value(); }

// STDOUT: # {{[0-9]+}} {{.*}}preprocess.cpp
// STDOUT-DAG: plain_old_function
// STDOUT-DAG: cudaq_test_dep_value

// FILE: # {{[0-9]+}} {{.*}}preprocess.cpp
// FILE-DAG: plain_old_function
// FILE-DAG: cudaq_test_dep_value

// ORCA: # {{[0-9]+}} {{.*}}preprocess.cpp
// ORCA-DAG: plain_old_function
// ORCA-DAG: cudaq_test_dep_value
