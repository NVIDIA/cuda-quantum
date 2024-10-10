/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qis/qubit_qis.h"

namespace cudaq {
void entryPoint(const std::function<void(cudaq::qvector<> &)> &statePrep);

void qft(cudaq::qview<> qubits);
void another(cudaq::qview<> qubits, std::size_t);

} // namespace cudaq