/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

#include <cstring>
#include <gtest/gtest.h>
#include <vector>

using namespace cudaq::nvqlink;

// Test with raw buffers (simpler, no Channel dependency)
class StreamTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Allocate a simple buffer for testing
    buffer_memory.resize(2048);
  }

  std::vector<uint8_t> buffer_memory;
};

TEST_F(StreamTest, OutputStreamBasicTypes) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());

  // Write various types
  out.write(static_cast<uint32_t>(42));
  out.write(static_cast<uint64_t>(12345678901234ULL));
  out.write(3.14159f);
  out.write(2.71828);

  // Verify something was written
  EXPECT_GT(out.bytes_written(), 0);
}

TEST_F(StreamTest, InputStreamBasicTypes) {
  // First write data
  OutputStream out(buffer_memory.data(), buffer_memory.size());
  out.write(static_cast<uint32_t>(42));
  out.write(static_cast<uint64_t>(12345678901234ULL));
  out.write(3.14159f);
  out.write(2.71828);

  size_t bytes_written = out.bytes_written();

  // Now read it back
  InputStream in(buffer_memory.data(), bytes_written);

  EXPECT_EQ(in.read<uint32_t>(), 42);
  EXPECT_EQ(in.read<uint64_t>(), 12345678901234ULL);
  EXPECT_FLOAT_EQ(in.read<float>(), 3.14159f);
  EXPECT_DOUBLE_EQ(in.read<double>(), 2.71828);
}

TEST_F(StreamTest, RoundTripInteger) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());
  int value = 123456;
  out.write(value);

  size_t bytes_written = out.bytes_written();

  InputStream in(buffer_memory.data(), bytes_written);
  int read_value = in.read<int>();

  EXPECT_EQ(value, read_value);
}

TEST_F(StreamTest, RoundTripMultipleValues) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());

  uint8_t u8 = 255;
  uint16_t u16 = 65535;
  uint32_t u32 = 4294967295U;
  int32_t i32 = -12345;

  out.write(u8);
  out.write(u16);
  out.write(u32);
  out.write(i32);

  size_t bytes_written = out.bytes_written();
  InputStream in(buffer_memory.data(), bytes_written);

  EXPECT_EQ(in.read<uint8_t>(), u8);
  EXPECT_EQ(in.read<uint16_t>(), u16);
  EXPECT_EQ(in.read<uint32_t>(), u32);
  EXPECT_EQ(in.read<int32_t>(), i32);
}

TEST_F(StreamTest, RoundTripStruct) {
  struct TestData {
    uint32_t id;
    float value;
    double timestamp;
  };

  OutputStream out(buffer_memory.data(), buffer_memory.size());

  TestData data{42, 3.14f, 1234567890.0};
  out.write(data);

  size_t bytes_written = out.bytes_written();
  InputStream in(buffer_memory.data(), bytes_written);
  TestData read_data = in.read<TestData>();

  EXPECT_EQ(read_data.id, data.id);
  EXPECT_FLOAT_EQ(read_data.value, data.value);
  EXPECT_DOUBLE_EQ(read_data.timestamp, data.timestamp);
}

TEST_F(StreamTest, RoundTripMultipleStructs) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());

  // Write multiple values in sequence
  for (uint32_t i = 0; i < 5; ++i) {
    out.write(i);
  }

  size_t bytes_written = out.bytes_written();
  InputStream in(buffer_memory.data(), bytes_written);

  // Read them back
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(in.read<uint32_t>(), i);
  }
}

TEST_F(StreamTest, BytesReadTracking) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());
  out.write(static_cast<uint32_t>(42));
  out.write(static_cast<uint64_t>(100));

  size_t expected_size = sizeof(uint32_t) + sizeof(uint64_t);
  EXPECT_EQ(out.bytes_written(), expected_size);

  InputStream in(buffer_memory.data(), expected_size);
  EXPECT_EQ(in.bytes_read(), 0);

  // Read one value
  in.read<uint32_t>();
  EXPECT_EQ(in.bytes_read(), sizeof(uint32_t));

  // Read second value
  in.read<uint64_t>();
  EXPECT_EQ(in.bytes_read(), expected_size);
}

TEST_F(StreamTest, RemainingCapacity) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());

  size_t initial_capacity = out.remaining_capacity();
  EXPECT_EQ(initial_capacity, buffer_memory.size());

  out.write(static_cast<uint64_t>(42));

  size_t after_write = out.remaining_capacity();
  EXPECT_EQ(after_write, initial_capacity - sizeof(uint64_t));
}

TEST_F(StreamTest, WriteToCapacity) {
  OutputStream out(buffer_memory.data(), buffer_memory.size());

  // Fill buffer with uint64_t values
  size_t capacity = out.remaining_capacity();
  size_t num_values = capacity / sizeof(uint64_t);

  for (size_t i = 0; i < num_values; ++i) {
    out.write(static_cast<uint64_t>(i));
  }

  EXPECT_LE(out.bytes_written(), buffer_memory.size());
  EXPECT_GT(out.bytes_written(), 0);
}
