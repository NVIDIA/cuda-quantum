/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatUtils.h"
#include <CuDensityMatErrorHandling.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cudaq::dynamics;

namespace {
// Fills a device buffer and reads it back.
bool roundTrips(void *devicePtr, std::size_t sizeBytes) {
  std::vector<unsigned char> written(sizeBytes), read(sizeBytes);
  for (std::size_t i = 0; i < sizeBytes; ++i)
    written[i] = static_cast<unsigned char>(i * 31 + 7);
  HANDLE_CUDA_ERROR(
      cudaMemcpy(devicePtr, written.data(), sizeBytes, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(read.data(), devicePtr, sizeBytes, cudaMemcpyDeviceToHost));
  return written == read;
}
} // namespace

TEST(GpuFabricOptionTest, DomainSizeForNamedFabrics) {
  EXPECT_EQ(detail::gpuFabricDomainSize("MNNVL", 16, 4), 16);
  EXPECT_EQ(detail::gpuFabricDomainSize("mnnvl", 16, 4), 16);
  EXPECT_EQ(detail::gpuFabricDomainSize("NVL", 16, 4), 4);
  EXPECT_EQ(detail::gpuFabricDomainSize("Nvl", 16, 4), 4);
  EXPECT_EQ(detail::gpuFabricDomainSize("NONE", 16, 4), 1);
  EXPECT_EQ(detail::gpuFabricDomainSize("none", 16, 4), 1);
}

TEST(GpuFabricOptionTest, DomainSizeForIntegers) {
  EXPECT_EQ(detail::gpuFabricDomainSize("1", 16, 4), 1);
  EXPECT_EQ(detail::gpuFabricDomainSize("8", 16, 4), 8);
  EXPECT_EQ(detail::gpuFabricDomainSize("72", 16, 4), 72);
}

TEST(GpuFabricOptionTest, InvalidValuesThrow) {
  for (const char *value : {"", "0", "-1", "4x", " 4", "abc", "MNNVL2"})
    EXPECT_THROW(detail::gpuFabricDomainSize(value, 16, 4),
                 std::invalid_argument)
        << "CUDAQ_GPU_FABRIC=" << value;
}

TEST(GpuFabricOptionTest, UnsetDoesNotRequestFabricMemory) {
  EXPECT_FALSE(detail::requestsFabricMemory(nullptr, 1, 1));
  EXPECT_FALSE(detail::requestsFabricMemory(nullptr, 8, 4));
}

TEST(GpuFabricOptionTest, SingleNodeNeverRequestsFabricMemory) {
  for (const char *value : {"MNNVL", "NVL", "NONE", "4", "64"}) {
    EXPECT_FALSE(detail::requestsFabricMemory(value, 1, 1)) << value;
    EXPECT_FALSE(detail::requestsFabricMemory(value, 4, 4)) << value;
  }
}

TEST(GpuFabricOptionTest, MultiNodeRequiresDomainSpanningAllRanks) {
  EXPECT_TRUE(detail::requestsFabricMemory("MNNVL", 8, 4));
  EXPECT_TRUE(detail::requestsFabricMemory("8", 8, 4));
  EXPECT_TRUE(detail::requestsFabricMemory("16", 8, 4));
  EXPECT_FALSE(detail::requestsFabricMemory("NVL", 8, 4));
  EXPECT_FALSE(detail::requestsFabricMemory("NONE", 8, 4));
  EXPECT_FALSE(detail::requestsFabricMemory("4", 8, 4));
}

TEST(GpuFabricOptionTest, InvalidValuesThrowOnAnyTopology) {
  EXPECT_THROW(detail::requestsFabricMemory("bogus", 1, 1),
               std::invalid_argument);
  EXPECT_THROW(detail::requestsFabricMemory("bogus", 4, 4),
               std::invalid_argument);
  EXPECT_THROW(detail::requestsFabricMemory("bogus", 8, 4),
               std::invalid_argument);
}

TEST(MpiBufferTest, ReservesOnlyToGrow) {
  MpiBuffer buffer;
  EXPECT_EQ(buffer.sizeBytes(), 0);
  void *first = buffer.reserve(1024, /*useFabricMemory=*/false);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(buffer.sizeBytes(), 1024);
  EXPECT_FALSE(buffer.isFabricMemory());

  EXPECT_EQ(buffer.reserve(512, false), first);
  EXPECT_EQ(buffer.reserve(1024, false), first);
  EXPECT_EQ(buffer.sizeBytes(), 1024);

  void *grown = buffer.reserve(4096, false);
  ASSERT_NE(grown, nullptr);
  EXPECT_EQ(buffer.sizeBytes(), 4096);
  EXPECT_TRUE(roundTrips(grown, 4096));
}

TEST(MpiBufferTest, ReleaseFreesNonFabricMemory) {
  MpiBuffer buffer;
  ASSERT_NE(buffer.reserve(1024, false), nullptr);
  buffer.release();
  EXPECT_EQ(buffer.sizeBytes(), 0);
  ASSERT_NE(buffer.reserve(256, false), nullptr);
  EXPECT_EQ(buffer.sizeBytes(), 256);
}

TEST(MpiBufferTest, ResetIsIdempotent) {
  MpiBuffer buffer;
  ASSERT_NE(buffer.reserve(1024, false), nullptr);
  buffer.reset();
  EXPECT_EQ(buffer.sizeBytes(), 0);
  EXPECT_NO_THROW(buffer.reset());
  EXPECT_EQ(buffer.sizeBytes(), 0);
}

// Fabric memory needs a fabric-capable driver and an IMEX channel, which most
// test systems lack; there the request falls back to `cudaMalloc`. Either way
// the buffer must be usable.
TEST(MpiBufferTest, FabricRequestYieldsUsableMemory) {
  const auto unsupported = DeviceAllocator::testFabricAllocation();
  std::cout << "Fabric memory: "
            << (unsupported ? "unavailable (" + *unsupported + ")"
                            : std::string("available"))
            << "\n";

  constexpr std::size_t sizeBytes = 1 << 20;
  MpiBuffer buffer;
  void *data = buffer.reserve(sizeBytes, /*useFabricMemory=*/true);
  ASSERT_NE(data, nullptr);
  EXPECT_GE(buffer.sizeBytes(), sizeBytes);
  EXPECT_EQ(buffer.isFabricMemory(), !unsupported.has_value());
  EXPECT_TRUE(roundTrips(data, sizeBytes));

  buffer.release();
  if (buffer.isFabricMemory()) {
    // Fabric memory is kept across uses.
    EXPECT_GE(buffer.sizeBytes(), sizeBytes);
    EXPECT_EQ(buffer.reserve(sizeBytes / 2, true), data);
  } else {
    EXPECT_EQ(buffer.sizeBytes(), 0);
  }
}
