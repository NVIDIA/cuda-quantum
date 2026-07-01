/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: device-call-realtime-lib

// clang-format off
// Compile one CUDA-Q app and exercise shared-memory and host-dispatch
// device_call paths through the co-linked service library.
// Use qpp-cpu explicitly because the custatevec simulator calls
// cudaDeviceSynchronize(), which conflicts with the persistent dispatch kernel.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ --target qpp-cpu -frealtime-lowering -I%S/Inputs %s %cudaq_lib_dir/%cudaq_shlibprefixcudaq-test-device-call-realtime-scalar-gpu-dispatch%cudaq_shlibext -o %t/app-gpu
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app-gpu --cudaq-device-call=shared-memory | FileCheck --check-prefix=SHM %s
// RUN: cd %t && nvq++ --target qpp-cpu -frealtime-lowering -I%S/Inputs %s %cudaq_lib_dir/%cudaq_shlibprefixcudaq-test-device-call-realtime-scalar-host-dispatch%cudaq_shlibext -o %t/app-host
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app-host --cudaq-device-call=host-dispatch | FileCheck --check-prefix=HOST %s
// clang-format on

#include <cudaq.h>

#include <cstdio>

extern "C" int addThem(int a, int b);
extern "C" float multiplyFloats(float a, float b);

__qpu__ int kernel(int a, int b) {
  return cudaq::device_call(0, addThem, a, b);
}

__qpu__ float floatKernel(float a, float b) {
  return cudaq::device_call(0, multiplyFloats, a, b);
}

__qpu__ int measuredIntegerKernel() {
  cudaq::qvector q(3);
  x(q[0]);
  x(q[2]);
  auto measured = cudaq::to_integer(cudaq::to_bools(mz(q)));
  return cudaq::device_call(0, addThem, static_cast<int>(measured), 37);
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);

  auto results = cudaq::run(1, kernel, 19, 23);
  int value = results.front();
  std::printf("device_call int result = %d\n", value);

  auto floatResults = cudaq::run(1, floatKernel, 6.0f, 7.0f);
  float floatValue = floatResults.front();
  std::printf("device_call float result = %.1f\n", floatValue);

  auto measuredResults = cudaq::run(1, measuredIntegerKernel);
  int measuredValue = measuredResults.front();
  std::printf("device_call measured integer result = %d\n", measuredValue);

  cudaq::realtime::finalize();
  return value == 42 && floatValue == 42.0f && measuredValue == 42 ? 0 : 1;
}

// SHM: device_call int result = 42
// SHM: device_call float result = 42.0
// SHM: device_call measured integer result = 42
// HOST: device_call int result = 42
// HOST: device_call float result = 42.0
// HOST: device_call measured integer result = 42
