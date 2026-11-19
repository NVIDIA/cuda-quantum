/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qubit_qis.h"

// Kernels defined in separate source files, so the Quake registered for
// `call_flip` only declares `flip`.
namespace interop_test {
__qpu__ void flip(cudaq::qview<> qubits, std::size_t i);
__qpu__ void call_flip(cudaq::qview<> qubits, std::size_t i);
} // namespace interop_test
