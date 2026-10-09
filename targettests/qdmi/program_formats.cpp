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
// RUN: env CUDAQ_LOG_LEVEL=info %t.auto | FileCheck --check-prefixes=RESULT,AUTO %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm2 %s -o %t.qasm2
// RUN: env CUDAQ_LOG_LEVEL=info %t.qasm2 | FileCheck --check-prefixes=RESULT,QASM2 %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm2-braket %s -o %t.qasm2-braket
// RUN: env CUDAQ_LOG_LEVEL=info %t.qasm2-braket | FileCheck --check-prefixes=RESULT,QASM2-BRAKET %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm3 %s -o %t.qasm3
// RUN: env CUDAQ_LOG_LEVEL=info %t.qasm3 | FileCheck --check-prefixes=RESULT,QASM3 %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-base-module %s -o %t.qir-base-module
// RUN: env CUDAQ_LOG_LEVEL=info %t.qir-base-module | FileCheck --check-prefixes=RESULT,QIR-BASE-MODULE %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-base-string %s -o %t.qir-base-string
// RUN: env CUDAQ_LOG_LEVEL=info %t.qir-base-string | FileCheck --check-prefixes=RESULT,QIR-BASE-STRING %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-adaptive-module %s -o %t.qir-adaptive-module
// RUN: env CUDAQ_LOG_LEVEL=info %t.qir-adaptive-module | FileCheck --check-prefixes=RESULT,QIR-ADAPTIVE-MODULE %s
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qir-adaptive-string %s -o %t.qir-adaptive-string
// RUN: env CUDAQ_LOG_LEVEL=info %t.qir-adaptive-string | FileCheck --check-prefixes=RESULT,QIR-ADAPTIVE-STRING %s
// clang-format on

#include <cudaq.h>
#include <iostream>

struct simple_x {
  void operator()() __qpu__ {
    cudaq::qvector qubits(3);
    x(qubits[0]);
    mz(qubits);
  }
};

int main() {
  const auto result = cudaq::sample(32, simple_x{});
  std::cout << "ones=" << result.count("100") << '\n';
}

// AUTO: through 'qir-adaptive-module' transport.
// QASM2: through 'qasm2' transport.
// QASM2-BRAKET: through 'qasm2-braket' transport.
// QASM3: through 'qasm3' transport.
// QIR-BASE-MODULE: through 'qir-base-module' transport.
// QIR-BASE-STRING: through 'qir-base-string' transport.
// QIR-ADAPTIVE-MODULE: through 'qir-adaptive-module' transport.
// QIR-ADAPTIVE-STRING: through 'qir-adaptive-string' transport.
// RESULT: ones=32
