/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that nvq++ forwards -fquantum-opt-level to cudaq-quake.

// RUN: nvq++ -c %s -v -fquantum-opt-level=2 -o %t.o | FileCheck %s

// CHECK: cudaq-quake {{.*}}--quantum-opt-level=2

#include "quantum_opt_level.cpp"
