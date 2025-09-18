/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/qclink/qclink.h"
#include "test_config.h"

using namespace cudaq::qclink;

TEST(QCLinkCudaChannelTester, checkChannelMemoryWorks) {
  cpu_shmem_device channel;
  channel.connect();
  int value = 42, getBack = 0;
  auto devPtr = channel.malloc(sizeof(int));
  channel.send(devPtr, &value);
  channel.recv(&getBack, devPtr);
  EXPECT_EQ(getBack, value);
  channel.free(devPtr);
  channel.disconnect();
}

TEST(QCLinkCudaChannelTester, checkCallbacks) {
  // Whoa, this is confusing...
  // Mapping device function library path -> vector<device_function>, where
  // the device_function is composed of a function name and
  // an optional unmarshaller (which we'll use here in this test,
  // don't have MLIR unmarshaller here)
  std::unordered_map<std::string, std::vector<device::device_function>>
      devcallbacks{{CUDAQ_TEST_BINARY_DIR "/unittests/libshmem-add.so",
                    {{device::device_function{
                        "add", [](void *sym, device_ptr &result,
                                  const std::vector<device_ptr> &args) {
                          // Here we know the function symbol signature
                          auto func = reinterpret_cast<int (*)(int, int)>(sym);
                          // We know how shmem_channel stores device_ptrs
                          int i = *reinterpret_cast<int *>(args[0].handle),
                              j = *reinterpret_cast<int *>(args[1].handle);
                          // Call, get the result
                          auto res = func(i, j);
                          std::memcpy((void *)result.handle, &res, 4);
                        }}}}}};

  // Create the channel and connect
  cpu_shmem_device channel(devcallbacks);
  channel.connect();

  // Create input arguments, and a place to store the result
  int i = 33, j = 22, k = 0;
  auto devPI = channel.malloc(4);
  auto devPJ = channel.malloc(4);
  auto devPK = channel.malloc(4);
  channel.send(devPI, &i);
  channel.send(devPJ, &j);

  // Launch teh callback
  channel.launch_callback("add", devPK, {devPI, devPJ});

  // Get back the result.
  channel.recv(&k, devPK);

  EXPECT_EQ(k, 33 + 22);
}
