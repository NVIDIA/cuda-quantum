/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DeviceCallLibrary.h"

#include <cstdint>
#include <cstring>

// Realtime lowering retains references to the logical call targets. These
// link-only definitions must never execute because dispatch uses the raw
// handlers registered below.
extern "C" __device__ int nativeCountPackedBits(const bool *, std::uint64_t,
                                                std::uint64_t) {
  __trap();
  return 0;
}

extern "C" __device__ void nativeIsEven(bool *, std::uint64_t, const int *,
                                        std::uint64_t) {
  __trap();
}

namespace {

__device__ std::int32_t nativeCountPackedBitsHandler(const void *input,
                                                     void *output,
                                                     std::uint32_t argLen,
                                                     std::uint32_t maxResultLen,
                                                     std::uint32_t *resultLen) {
  if (!input || !output || !resultLen || maxResultLen < sizeof(std::int32_t) ||
      argLen < sizeof(std::uint64_t))
    return -1;

  // The `(vector<bool>, uint64_t) -> int` signature marshals a bit count and
  // packed bytes, followed by alignment padding and the scalar bias.
  const auto *const bytes = static_cast<const std::uint8_t *>(input);
  const auto bitCount = *reinterpret_cast<const std::uint64_t *>(bytes);
  std::uint64_t offset = sizeof(std::uint64_t);
  const std::uint64_t packedBytes = bitCount / 8 + (bitCount % 8 != 0);
  if (packedBytes > argLen - offset)
    return -1;

  std::int32_t total = 0;
  // Compute the population count directly from the LSB-first packed payload
  // without first unpacking it into a Boolean array.
  for (std::uint64_t i = 0; i < bitCount; ++i)
    total += (bytes[offset + i / 8] >> (i % 8)) & 1u;
  offset += packedBytes;
  offset = (offset + 7) & ~std::uint64_t{7};
  if (offset > argLen || sizeof(std::uint64_t) > argLen - offset)
    return -1;

  const auto bias = *reinterpret_cast<const std::uint64_t *>(bytes + offset);
  total += bias;
  offset += sizeof(std::uint64_t);
  if (offset != argLen)
    return -1;

  std::memcpy(output, &total, sizeof(total));
  *resultLen = sizeof(total);
  return 0;
}

__device__ std::int32_t nativeIsEvenHandler(const void *input, void *output,
                                            std::uint32_t argLen,
                                            std::uint32_t maxResultLen,
                                            std::uint32_t *resultLen) {
  if (!input || !resultLen || argLen < 2 * sizeof(std::uint64_t))
    return -1;

  // The `(vector<bool>&, const vector<int>&) -> void` signature marshals the
  // output and input counts followed by the integer values. The response
  // contains the output Boolean vector as packed bytes.
  const auto *const bytes = static_cast<const std::uint8_t *>(input);
  const auto resultCount = *reinterpret_cast<const std::uint64_t *>(bytes);
  const auto valueCount =
      *reinterpret_cast<const std::uint64_t *>(bytes + sizeof(std::uint64_t));
  std::uint64_t offset = 2 * sizeof(std::uint64_t);
  if (resultCount != valueCount || valueCount > (argLen - offset) / sizeof(int))
    return -1;
  offset += valueCount * sizeof(int);
  if (offset != argLen)
    return -1;

  const std::uint64_t packedBytes = resultCount / 8 + (resultCount % 8 != 0);
  if (packedBytes > maxResultLen || (packedBytes && !output))
    return -1;

  auto *const packed = static_cast<std::uint8_t *>(output);
  for (std::uint64_t i = 0; i < packedBytes; ++i)
    packed[i] = 0;
  const auto *const values =
      reinterpret_cast<const int *>(bytes + 2 * sizeof(std::uint64_t));
  // Check each integer for evenness and set its corresponding LSB-first bit in
  // the packed Boolean result.
  for (std::uint64_t i = 0; i < valueCount; ++i)
    if (values[i] % 2 == 0)
      packed[i / 8] |= std::uint8_t{1} << (i % 8);

  *resultLen = static_cast<std::uint32_t>(packedBytes);
  return 0;
}

__global__ void initNativeBitpackedTable(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  auto &entry = entries[0];
  cudaq_internal::device_call::detail::zeroObject(entry);
  entry.handler.device_fn_ptr =
      reinterpret_cast<void *>(&nativeCountPackedBitsHandler);
  entry.function_id = cudaq::realtime::fnv1a_hash("nativeCountPackedBits");
  entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entry.schema.num_args = 2;
  entry.schema.args[0] = {CUDAQ_TYPE_BIT_PACKED, {0}, 0, 0};
  entry.schema.args[1] = {CUDAQ_TYPE_INT64, {0}, 8, 1};
  entry.schema.num_results = 1;
  entry.schema.results[0] = {CUDAQ_TYPE_INT32, {0}, 4, 1};

  auto &isEvenEntry = entries[1];
  cudaq_internal::device_call::detail::zeroObject(isEvenEntry);
  isEvenEntry.handler.device_fn_ptr =
      reinterpret_cast<void *>(&nativeIsEvenHandler);
  isEvenEntry.function_id = cudaq::realtime::fnv1a_hash("nativeIsEven");
  isEvenEntry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  isEvenEntry.schema.num_args = 1;
  isEvenEntry.schema.args[0] = {CUDAQ_TYPE_ARRAY_INT32, {0}, 0, 0};
  isEvenEntry.schema.num_results = 1;
  isEvenEntry.schema.results[0] = {CUDAQ_TYPE_BIT_PACKED, {0}, 0, 0};
}

using NativeBitpackedService =
    cudaq_internal::device_call::detail::GeneratedDeviceCallService<
        &initNativeBitpackedTable, 2>;

cudaq::realtime::DeviceCallService *getNativeBitpackedService() {
  return NativeBitpackedService::getService();
}

} // namespace

extern "C" cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"native-bitpacked", &getNativeBitpackedService};
}
