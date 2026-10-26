/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "device_call_realtime_lib.cuh"

#include "DeviceCallLibrary.h"

constexpr auto countTrueBitsSchema =
    cudaq_internal::device_call::detail::makeHandlerSchema<&countTrueBits>();
static_assert(countTrueBitsSchema.num_args == 2);
static_assert(countTrueBitsSchema.args[0].type_id == CUDAQ_TYPE_BIT_PACKED);
static_assert(countTrueBitsSchema.args[1].type_id == CUDAQ_TYPE_INT64);
static_assert(countTrueBitsSchema.num_results == 1);
static_assert(countTrueBitsSchema.results[0].type_id == CUDAQ_TYPE_INT32);

constexpr auto sumByteVectorSchema =
    cudaq_internal::device_call::detail::makeHandlerSchema<&sumByteVector>();
static_assert(sumByteVectorSchema.num_args == 2);
static_assert(sumByteVectorSchema.args[0].type_id == CUDAQ_TYPE_ARRAY_UINT8);
static_assert(sumByteVectorSchema.args[1].type_id == CUDAQ_TYPE_INT32);
static_assert(sumByteVectorSchema.num_results == 1);
static_assert(sumByteVectorSchema.results[0].type_id == CUDAQ_TYPE_INT32);

constexpr auto integerToBinaryVectorSchema =
    cudaq_internal::device_call::detail::makeHandlerSchema<
        &integerToBinaryVector>();
static_assert(integerToBinaryVectorSchema.num_args == 1);
static_assert(integerToBinaryVectorSchema.args[0].type_id == CUDAQ_TYPE_INT64);
static_assert(integerToBinaryVectorSchema.num_results == 1);
static_assert(integerToBinaryVectorSchema.results[0].type_id ==
              CUDAQ_TYPE_BIT_PACKED);

CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(test_device_call_array)
CUDAQ_DEVICE_CALL_EXPORT(countTrueBits)
CUDAQ_DEVICE_CALL_EXPORT(countTrueMeasures)
CUDAQ_DEVICE_CALL_EXPORT(sumIntVector)
CUDAQ_DEVICE_CALL_EXPORT(sumByteVector)
CUDAQ_DEVICE_CALL_EXPORT(incrementIntVector)
CUDAQ_DEVICE_CALL_EXPORT(integerToBinaryVector)
CUDAQ_DEVICE_CALL_EXPORT(sumFloatVector)
CUDAQ_DEVICE_CALL_LIBRARY_END()
