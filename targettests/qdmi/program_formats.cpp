/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: qdmi
// clang-format off
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default %s -o %t.auto
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.auto.bc %t.auto | FileCheck --check-prefix=RESULT %s
// RUN: llvm-dis %t.auto.bc -o - | FileCheck --check-prefix=QIR %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm2 %s -o %t.qasm2
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qasm2.txt %t.qasm2 | FileCheck --check-prefix=RESULT %s
// RUN: FileCheck --check-prefix=QASM %s < %t.qasm2.txt
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm3 %s -o %t.qasm3
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qasm3.txt %t.qasm3 | FileCheck --check-prefix=RESULT %s
// RUN: cmp %t.qasm2.txt %t.qasm3.txt
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-base-module %s -o %t.qir-base-module
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qir-base.bc %t.qir-base-module | FileCheck --check-prefix=RESULT %s
// RUN: llvm-dis %t.qir-base.bc -o - | FileCheck --check-prefix=QIR %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-base-string %s -o %t.qir-base-string
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qir-base.ll %t.qir-base-string | FileCheck --check-prefix=RESULT %s
// RUN: FileCheck --check-prefix=QIR %s < %t.qir-base.ll
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-adaptive-module %s -o %t.qir-adaptive-module
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qir-adaptive.bc %t.qir-adaptive-module | FileCheck --check-prefix=RESULT %s
// RUN: llvm-dis %t.qir-adaptive.bc -o - | FileCheck --check-prefix=QIR %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-adaptive-string %s -o %t.qir-adaptive-string
// RUN: env CUDAQ_QDMI_DUMP_PROGRAM=%t.qir-adaptive.ll %t.qir-adaptive-string | FileCheck --check-prefix=RESULT %s
// RUN: FileCheck --check-prefix=QIR %s < %t.qir-adaptive.ll
// clang-format on

#include <cudaq.h>
#include <iostream>

struct simple_x {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    x(qubit);
    mz(qubit);
  }
};

int main() {
  const auto result = cudaq::sample(32, simple_x{});
  std::cout << "ones=" << result.count("1") << '\n';
}

// RESULT: ones=32
// QASM: OPENQASM 2.0;
// QIR: define
// QIR: @__quantum__qis__x
