/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {
// Helper to retrieve the number of GPU.
// It works with or without CUDA dependency.
// If CUDA is present, returns the actual number of GPU devices. Otherwise,
// returns 0.
int getCudaDeviceCount();

// Helper to set the current CUDA device for the calling thread.
void setCudaDevice(int deviceId);
} // namespace cudaq
