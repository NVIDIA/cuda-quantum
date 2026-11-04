/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cross_file.h"

__qpu__ void interop_test::flip(cudaq::qview<> qubits, std::size_t i) {
  x(qubits[i]);
}
