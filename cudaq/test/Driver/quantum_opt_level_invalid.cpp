/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that nvq++ rejects an out-of-range -fquantum-opt-level.

// RUN: not nvq++ -c %s -fquantum-opt-level=4 -o %t.o 2>&1 | FileCheck %s

// CHECK: invalid -fquantum-opt-level=4

#include "quantum_opt_level.cpp"
