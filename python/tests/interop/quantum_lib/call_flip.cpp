/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cross_file.h"

// Keep this out of flip.cpp, so that `flip` stays a declaration here.
__qpu__ void interop_test::call_flip(cudaq::qview<> qubits, std::size_t i) {
  flip(qubits, i);
}
