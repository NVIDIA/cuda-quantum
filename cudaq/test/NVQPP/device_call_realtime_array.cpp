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
// device_call paths for std::vector payloads.
// Use qpp-cpu explicitly because the custatevec simulator calls
// cudaDeviceSynchronize(), which conflicts with the persistent dispatch kernel.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ --target qpp-cpu -frealtime-lowering -I%S/Inputs %s %cudaq_lib_dir/%cudaq_shlibprefixcudaq-test-device-call-realtime-array-gpu-dispatch%cudaq_shlibext -o %t/app-gpu
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app-gpu --cudaq-device-call=shared-memory | FileCheck --check-prefix=SHM %s
// RUN: cd %t && nvq++ --target qpp-cpu -frealtime-lowering -I%S/Inputs %s %cudaq_lib_dir/%cudaq_shlibprefixcudaq-test-device-call-realtime-array-host-dispatch%cudaq_shlibext -o %t/app-host
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app-host --cudaq-device-call=host-dispatch | FileCheck --check-prefix=HOST %s
// clang-format on

#include <cudaq.h>

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int countTrueBits(const std::vector<bool> &bits,
                             std::uint64_t bias);
extern "C" int countTrueMeasures(
    const std::vector<cudaq::measure_result> &bits, std::uint64_t bias);
extern "C" int sumIntVector(const std::vector<int> &values, int bias);
extern "C" void incrementIntVector(std::vector<int> &out,
                                   const std::vector<int> &values, int delta);
extern "C" void integerToBinaryVector(std::vector<bool> &out,
                                       std::uint64_t value);
extern "C" float sumFloatVector(const std::vector<float> &values, float bias);

__qpu__ int boolVectorKernel(std::vector<bool> bits) {
  return cudaq::device_call(0, countTrueBits, bits, std::uint64_t{15});
}

__qpu__ int measureVectorKernel() {
  cudaq::qvector q(6);
  x(q[0]);
  x(q[2]);
  x(q[3]);
  x(q[5]);
  auto bits = mz(q);
  return cudaq::device_call(0, countTrueMeasures, bits, std::uint64_t{15});
}

__qpu__ int intVectorKernel(std::vector<int> values) {
  return cudaq::device_call(0, sumIntVector, values, 24);
}

__qpu__ std::vector<int> incrementIntVectorKernel(std::vector<int> values) {
  std::vector<int> out(values.size());
  cudaq::device_call(0, incrementIntVector, out, values, 1);
  return out;
}

__qpu__ std::vector<bool> binaryVectorKernel(std::uint64_t value) {
  std::vector<bool> out(6);
  cudaq::device_call(0, integerToBinaryVector, out, value);
  return out;
}

__qpu__ float floatVectorKernel(std::vector<float> values) {
  return cudaq::device_call(0, sumFloatVector, values, 35.0f);
}

static bool runInputStdvecTests() {
  std::vector<bool> bits = {true, false, true, true, false, true};
  auto boolResults = cudaq::run(1, boolVectorKernel, bits);
  int boolValue = boolResults.front();
  std::printf("device_call bool stdvec input result = %d\n", boolValue);

  int measureValue = measureVectorKernel();
  std::printf("device_call measure stdvec input result = %d\n", measureValue);

  std::vector<int> intValues = {3, 4, 5, 6};
  int intValue = intVectorKernel(intValues);
  std::printf("device_call i32 stdvec input result = %d\n", intValue);

  std::vector<float> floatValues = {1.5f, 2.5f, 3.0f};
  auto floatResults = cudaq::run(1, floatVectorKernel, floatValues);
  float floatValue = floatResults.front();
  std::printf("device_call f32 stdvec input result = %.1f\n", floatValue);

  return boolValue == 19 && measureValue == 19 && intValue == 42 &&
         floatValue == 42.0f;
}

static bool runByRefOutputStdvecTests() {
  std::vector<int> intValues = {3, 4, 5, 6};
  auto incrementedRuns = cudaq::run(1, incrementIntVectorKernel, intValues);
  auto incremented = incrementedRuns.front();
  std::printf("device_call i32 stdvec by-ref output =");
  for (auto value : incremented)
    std::printf(" %d", value);
  std::printf("\n");

  auto binaryRuns = cudaq::run(1, binaryVectorKernel, std::uint64_t{13});
  auto bits = binaryRuns.front();
  std::printf("device_call bool stdvec by-ref output =");
  for (bool bit : bits)
    std::printf(" %d", bit ? 1 : 0);
  std::printf("\n");

  return incremented.size() == 4 && incremented[0] == 4 &&
         incremented[1] == 5 && incremented[2] == 6 && incremented[3] == 7 &&
         bits.size() == 6 && bits[0] && !bits[1] && bits[2] && bits[3] &&
         !bits[4] && !bits[5];
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);
  bool passed = runInputStdvecTests() && runByRefOutputStdvecTests();

  cudaq::realtime::finalize();
  return passed ? 0 : 1;
}

// SHM: device_call bool stdvec input result = 19
// SHM: device_call measure stdvec input result = 19
// SHM: device_call i32 stdvec input result = 42
// SHM: device_call f32 stdvec input result = 42.0
// SHM: device_call i32 stdvec by-ref output = 4 5 6 7
// SHM: device_call bool stdvec by-ref output = 1 0 1 1 0 0
// HOST: device_call bool stdvec input result = 19
// HOST: device_call measure stdvec input result = 19
// HOST: device_call i32 stdvec input result = 42
// HOST: device_call f32 stdvec input result = 42.0
// HOST: device_call i32 stdvec by-ref output = 4 5 6 7
// HOST: device_call bool stdvec by-ref output = 1 0 1 1 0 0
