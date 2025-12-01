/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq.h"
#include "cudaq/nvqlink.h"

// Quantum Kernel, compiled with
// cudaq-quake quantum.cpp | \
//    cudaq-opt --memtoreg --load-cudaq-plugin lib/DeviceCallShmem.so \
//      --device-call-shmem --canonicalize --kernel-execution | \
//    cudaq-translate | \
//    llc -o quantum.o -filetype=obj --relocation-model=pic
//
// Note the custom plugin DeviceCallShmem.so. This plugin
// provides a cudaq-opt pass that transforms cc.device_call to an
// intrinsic library call in cudaq-nvqlink, which relies on the
// nvqlink real-time host / controller to mediate data marshaling and
// function callback on the specified device.
extern int add_q_kernel(int i, int j);

// Above kernel calls a device function from a libadd.so library,
// compiled with
// nvq++ add.cpp -shared -fPIC -o libadd.so

// The entire application can now be compiled and run as follows
// clang-format off
// nvq++ cpu_shmem_device_call_add.cpp quantum.o -I nvqlink/include -L nvqlink/lib -lcudaq-nvqlink -Wl,-rpath,nvqlink/lib
// ./a.out
// clang-format on

using namespace cudaq;

nvqlink::device::device_function add_func = nvqlink::device::device_function{
    "add", [](void *sym, nvqlink::device_ptr &result,
              const std::vector<nvqlink::device_ptr> &args) {
      // Here we know the function symbol signature
      auto func = reinterpret_cast<int (*)(int, int)>(sym);
      // We know how shmem_channel stores device_ptrs
      int i = *reinterpret_cast<int *>(args[0].handle),
          j = *reinterpret_cast<int *>(args[1].handle);
      // Call, get the result
      auto res = func(i, j);
      std::memcpy((void *)result.handle, &res, 4);
    }};

int main() {

  // Here we manually specify the CPU Shmem device callback map.
  std::unordered_map<std::string, std::vector<nvqlink::device::device_function>>
      devcallbacks{{"libadd.so", {add_func}}};

  std::vector<std::unique_ptr<nvqlink::device>> devices;
  devices.emplace_back(
      std::make_unique<nvqlink::cpu_shmem_device>(devcallbacks));
  devices.emplace_back(std::make_unique<nvqlink::nv_simulation_device>());
  nvqlink::lqpu cfg(std::move(devices));

  // Initialize the library
  nvqlink::initialize(&cfg);

  printf("With devices, 3 + 4 = %d\n", add_q_kernel(3, 4));

  nvqlink::shutdown();
}
