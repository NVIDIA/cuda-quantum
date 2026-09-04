/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "device_call_realtime_lib.cuh"

#include "DeviceCallLibrary.h"

CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(test_device_call_scalar)
CUDAQ_DEVICE_CALL_EXPORT(addThem)
CUDAQ_DEVICE_CALL_EXPORT(multiplyFloats)
CUDAQ_DEVICE_CALL_HOST_DISPATCH_LIBRARY_END()
