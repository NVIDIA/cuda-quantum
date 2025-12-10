/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: nvq++ %cpp_std %s -o %t --target ionq --emulate && %t | FileCheck --check-prefix=IONQ %s
// RUN: nvq++ %cpp_std %s -o %t --target oqc --emulate && %t | FileCheck --check-prefix=OQC %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  x<cudaq::ctrl>(q0, q1);
  x<cudaq::ctrl>(q0, q2); // requires a swap(q0,q1) for OQC target but not IonQ
  mz(q0);
  mz(q1);
  mz(q2);
}

int main() {
  std::cout << cudaq::contrib::draw(foo) << '\n';

  return 0;
}

// IONQ:      ╭───╮          ╭────╮
// IONQ: q0 : ┤ x ├──●────●──┤ mz ├
// IONQ:      ├───┤╭─┴─╮  │  ├────┤
// IONQ: q1 : ┤ x ├┤ x ├──┼──┤ mz ├
// IONQ:      ╰───╯╰───╯╭─┴─╮├────┤
// IONQ: q2 : ──────────┤ x ├┤ mz ├
// IONQ:                ╰───╯╰────╯

// OQC:      ╭───╮        ╭────╮      
// OQC: q0 : ┤ x ├──●───╳─┤ mz ├──────
// OQC:      ├───┤╭─┴─╮ │ ╰────╯╭────╮
// OQC: q1 : ┤ x ├┤ x ├─╳───●───┤ mz ├
// OQC:      ╰───╯╰───╯   ╭─┴─╮ ├────┤
// OQC: q2 : ─────────────┤ x ├─┤ mz ├
// OQC:                   ╰───╯ ╰────╯
