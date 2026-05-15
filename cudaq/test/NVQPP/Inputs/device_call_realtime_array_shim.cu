/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "device_call_realtime_lib.cuh"

#include "DeviceCallLibrary.h"

CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(test_device_call_array)
CUDAQ_DEVICE_CALL_EXPORT(countTrueBits)
CUDAQ_DEVICE_CALL_EXPORT(countTrueMeasures)
CUDAQ_DEVICE_CALL_EXPORT(sumIntVector)
CUDAQ_DEVICE_CALL_EXPORT(incrementIntVector)
CUDAQ_DEVICE_CALL_EXPORT(integerToBinaryVector)
CUDAQ_DEVICE_CALL_EXPORT(sumFloatVector)
CUDAQ_DEVICE_CALL_LIBRARY_END()
