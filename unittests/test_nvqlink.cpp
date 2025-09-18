/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq/nvqlink/nvqlink.h"

using namespace cudaq;

TEST(NVQLinkTester, checkSimple) {
  // Configure the system
  std::vector<std::unique_ptr<nvqlink::device>> devs;
  // construct a base device
  devs.emplace_back(std::make_unique<nvqlink::cpu_shmem_device>());
  // add a concretely defined device
  devs.emplace_back(std::make_unique<nvqlink::cuda_device>());
  devs.emplace_back(std::make_unique<nvqlink::nv_simulation_device>());
  nvqlink::lqpu cfg(std::move(devs));

  EXPECT_EQ(cfg.get_num_devices(), 3);
  EXPECT_EQ(cfg.get_num_qcs_devices(), 1);

  {
    // Initialize
    nvqlink::initialize(&cfg);

    // do some data maninpulation
    auto devPtr = nvqlink::malloc(4, 0);
    int i = 22, j = 0;
    nvqlink::memcpy(devPtr, &i);
    nvqlink::memcpy(&j, devPtr);
    EXPECT_EQ(i, j);
    nvqlink::free(devPtr);

    // shutdonw
    nvqlink::shutdown();
  }

  {
    // Initialize with rt host specified
    nvqlink::initialize(&cfg);

    // do some data maninpulation
    auto devPtr = nvqlink::malloc(4, 0);
    int i = 22, j = 0;
    nvqlink::memcpy(devPtr, &i);
    nvqlink::memcpy(&j, devPtr);
    EXPECT_EQ(i, j);
    nvqlink::free(devPtr);

    // shutdonw
    nvqlink::shutdown();
  }
}

const std::string quake = R"#(
module attributes {cc.sizeof_string = 32 : i64, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.triple = "x86_64-unknown-linux-gnu", quake.mangled_name_map = {__nvqpp__mlirgen__function_test._Z4testii = "_Z4testii"}} {  
  func.func @__nvqpp__mlirgen__function_test._Z4testii(%arg0: i32, %arg1: i32) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %c33_i32 = arith.constant 33 : i32
    %0 = cc.alloca i32
    cc.store %arg0, %0 : !cc.ptr<i32>
    %1 = cc.alloca i32
    cc.store %arg1, %1 : !cc.ptr<i32>
    %2 = cc.load %1 : !cc.ptr<i32>
    %3 = arith.addi %2, %c33_i32 : i32
    %4 = cc.load %0 : !cc.ptr<i32>
    %5 = arith.addi %3, %4 : i32
    return %5 : i32
  }
  func.func @_Z4testii(%arg0: i32, %arg1: i32) -> i32 attributes {no_this} {
    %0 = cc.undef i32
    return %0 : i32
  }
})#";

TEST(NVQLinkTester, checkWorkflow) {

  auto runTest = [&]() {
    auto hdl = nvqlink::load_kernel(quake, "function_test._Z4testii");
    auto retPtr = nvqlink::malloc(sizeof(int));
    int i = 2;
    int j = 3;
    // device_ptr value reference
    device_ptr<int> iPtr(&i);
    device_ptr<int> jPtr(&j);
    nvqlink::launch_kernel(hdl, retPtr, {iPtr, jPtr});
    int ret;
    nvqlink::memcpy(&ret, retPtr);
    EXPECT_EQ(ret, 38);
    nvqlink::free(retPtr);
  };
  {
    std::vector<std::unique_ptr<nvqlink::device>> devs;
    devs.emplace_back(std::make_unique<nvqlink::nv_simulation_device>());
    nvqlink::lqpu cfg(std::move(devs));
    nvqlink::initialize(&cfg);
    runTest();
    nvqlink::shutdown();
  }

  {
    std::vector<std::unique_ptr<nvqlink::device>> devs;
    devs.emplace_back(std::make_unique<nvqlink::nv_simulation_device>());
    nvqlink::lqpu cfg(std::move(devs));
    nvqlink::initialize<nvqlink::nv_simulation_rt_host>(&cfg);
    runTest();
    nvqlink::shutdown();
  }
}
