/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: nvq++ -v %s -o %t.linked 2>&1 | FileCheck %s \
// RUN:   --check-prefix=ONE_SHOT --implicit-check-not=lower-to-cfg
// RUN: nvq++ -c -v %s -o %t.staged 2>&1 | FileCheck %s \
// RUN:   --check-prefix=STAGED --implicit-check-not=lower-to-cfg
// RUN: nvq++ -c -v -flower-to-cfg %s -o %t.with_cfg 2>&1 | FileCheck %s \
// RUN:   --check-prefix=WITH_CFG
// RUN: nvq++ -c -v -fno-lower-to-cfg %s -o %t.without_cfg 2>&1 | \
// RUN:   FileCheck %s --check-prefix=WITHOUT_CFG \
// RUN:   --implicit-check-not=lower-to-cfg

#include "cudaq.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  x(q);
}

int main() { return 0; }

// clang-format off
// ONE_SHOT: cudaq-opt
// ONE_SHOT-SAME: --pass-pipeline={{.*}}device-code-loader,expand-measurements,distributed-device-call
// ONE_SHOT: cudaq-translate --convert-to=qir:0.1

// STAGED: cudaq-opt
// STAGED-SAME: --pass-pipeline={{.*}}device-code-loader,expand-measurements,distributed-device-call
// STAGED: cudaq-translate --convert-to=qir:0.1

// WITH_CFG: cudaq-opt
// WITH_CFG-SAME: --pass-pipeline={{.*}}device-code-loader,expand-measurements,lower-to-cfg,distributed-device-call
// WITH_CFG: cudaq-translate --convert-to=qir:0.1

// WITHOUT_CFG: cudaq-opt
// WITHOUT_CFG-SAME: --pass-pipeline={{.*}}device-code-loader,expand-measurements,distributed-device-call
// WITHOUT_CFG: cudaq-translate --convert-to=qir:0.1
// clang-format on
