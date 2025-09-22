/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq.h"
#include "cudaq/driver/device.h"

extern "C" int add(int, int);

__qpu__ int add_q_kernel(int i, int j) { return cudaq::device_call(add, i, j); }
