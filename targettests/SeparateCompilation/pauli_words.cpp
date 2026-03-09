/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ --enable-mlir -c %t/pauli_word_display.cpp -o %t/pauli_word_display.o && \
// RUN: nvq++ --enable-mlir -c %t/pauli_wordle.cpp -o %t/pauli_wordle.o && \
// RUN: nvq++ --enable-mlir %t/pauli_word_display.o %t/pauli_wordle.o -o %t/pauli_wordle.out && \
// RUN: %t/pauli_wordle.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- pauli_word_display.cpp

#include <iostream>
#include <span>
#include <string>

extern "C" {
void display(std::span<char> x) {
  std::string s{x.data(), x.size()};
  std::cout << "pauli word: " << s << '\n';
}
}

//--- pauli_wordle.cpp

#include <cudaq.h>

// Fake host C++ signature that matches. Since this is called on the device side
// the pauli_word will have been converted to a span.
extern "C" void display(const cudaq::pauli_word &pw);

__qpu__ void kerny(std::vector<cudaq::pauli_word> arg) {
  display(arg[0]);
  display(arg[1]);
  display(arg[2]);
  display(arg[3]);
}

__qpu__ void kernub(cudaq::pauli_word arg) { display(arg); }

int main() {
  cudaq::pauli_word pw0 = "YYZ";
  kernub(pw0);

  cudaq::pauli_word pw1 = "ZIZ";
  cudaq::pauli_word pw2 = "XXXY";
  cudaq::pauli_word pw3 = "YIIII";
  std::vector<cudaq::pauli_word> vpw{pw0, pw1, pw2, pw3};
  kerny(vpw);
  return 0;
}

// CHECK: pauli word: YYZ
// CHECK: pauli word: YYZ
// CHECK: pauli word: ZIZ
// CHECK: pauli word: XXXY
// CHECK: pauli word: YIIII
