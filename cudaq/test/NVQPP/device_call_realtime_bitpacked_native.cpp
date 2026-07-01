/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: device-call-realtime-lib

// clang-format off
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ --target qpp-cpu -frealtime-lowering %s %cudaq_lib_dir/%cudaq_shlibprefixcudaq-test-device-call-realtime-bitpacked-native-gpu-dispatch%cudaq_shlibext -o %t/app
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app --cudaq-device-call=shared-memory | FileCheck %s
// clang-format on

#include <cudaq.h>

#include <cstdint>
#include <cstdio>
#include <vector>

// The native handler returns `bias + popcount(bits)`. For example, the
// original 10-bit case has six set bits and bias 15, so it returns 21.
extern "C" int nativeCountPackedBits(const std::vector<bool> &bits,
                                     std::uint64_t bias);

extern "C" void nativeIsEven(std::vector<bool> &result,
                             const std::vector<int> &values);

__qpu__ int nativeBitpackedKernel(std::vector<bool> bits, std::uint64_t bias) {
  return cudaq::device_call(0, nativeCountPackedBits, bits, bias);
}

__qpu__ std::vector<bool> nativeIsEvenKernel(std::vector<int> values) {
  std::vector<bool> result(values.size());
  cudaq::device_call(0, nativeIsEven, result, values);
  return result;
}

static bool runCase(const char *name, const std::vector<bool> &bits,
                    std::uint64_t bias, int expected) {
  const int result = cudaq::run(1, nativeBitpackedKernel, bits, bias).front();
  std::printf("native bit-packed %s result = %d\n", name, result);
  return result == expected;
}

static bool runIsEvenCase(const std::vector<int> &values,
                          const std::vector<bool> &expected) {
  const auto result = cudaq::run(1, nativeIsEvenKernel, values).front();
  std::printf("native bit-packed is-even output =");
  for (bool bit : result)
    std::printf(" %d", bit ? 1 : 0);
  std::printf("\n");
  return result == expected;
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);

  bool passed = true;
  passed &= runCase("empty", {}, 7, 7);
  passed &= runCase("byte-aligned",
                    {true, false, true, false, true, false, true, false}, 2, 6);
  passed &= runCase(
      "partial-byte",
      {true, false, true, true, false, true, false, false, true, true}, 15, 21);
  passed &= runCase("multi-byte-tail",
                    {true, true, false, false, true, false, true, false, true,
                     false, false, true, false, true, false, false, true},
                    100, 108);
  passed &= runIsEvenCase({-3, -2, -1, 0, 1, 2, 11, 12},
                          {false, true, false, true, false, true, false, true});
  passed &= runIsEvenCase(
      {0, 1, 2, 3, 4, -5, -6, 7, 8, 9},
      {true, false, true, false, true, false, true, false, true, false});

  cudaq::realtime::finalize();
  return passed ? 0 : 1;
}

// CHECK: native bit-packed empty result = 7
// CHECK: native bit-packed byte-aligned result = 6
// CHECK: native bit-packed partial-byte result = 21
// CHECK: native bit-packed multi-byte-tail result = 108
// CHECK: native bit-packed is-even output = 0 1 0 1 0 1 0 1
// CHECK: native bit-packed is-even output = 1 0 1 0 1 0 1 0 1 0
