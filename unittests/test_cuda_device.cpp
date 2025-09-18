/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/qclink.h"
#include "test_config.h"

using namespace cudaq::qclink;

TEST(QCLinkCudaChannelTester, checkChannelMemoryWorks) {
  cuda_device dev;
  dev.connect();
  int value = 42, getBack = 0;
  auto devPtr = dev.malloc(sizeof(int));
  dev.send(devPtr, &value);
  dev.recv(&getBack, devPtr);
  EXPECT_EQ(getBack, value);
  dev.free(devPtr);
  dev.disconnect();
}

TEST(QCLinkCudaChannelTester, checkCallbacks) {
  // Mapping device function library path -> vector<device_function>, where
  // the device_function is composed of a function name and
  // an optional unmarshaller (which we'll use here in this test,
  // don't have MLIR unmarshaller here)
  std::unordered_map<std::string, std::vector<device::device_function>>
      devcallbacks{{CUDAQ_TEST_BINARY_DIR "/unittests/CMakeFiles/vec-add.dir/"
                    "test_device_libraries/vec_add.fatbin",
                    {{device::device_function{"vectorAdd"}}}}};

  // Create the channel and connect
  cuda_device channel(devcallbacks);
  channel.connect();

  // Create input arguments, and a place to store the result
  int N = 100;
  std::vector<float> i(N, 5.), j(N, 6.), k(N);
  auto [devPI, devPJ, devPK] =
      channel.malloc(sizeof(float) * N, sizeof(float) * N, sizeof(float) * N);

  // set the data on the pointer across the channel
  channel.send(devPI, i.data());
  channel.send(devPJ, j.data());

  // Launch the callback
  channel.launch_callback("vectorAdd", {devPI, devPJ, devPK, &N});

  // Get back the result.
  channel.recv(k.data(), devPK);

  // check it all worked
  EXPECT_EQ(k[0], 11);

  // free the data
  channel.free(devPI, devPJ, devPK);
}
