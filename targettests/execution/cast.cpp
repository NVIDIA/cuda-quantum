/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target infleqtion      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ %cpp_std --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

struct testBool {
  auto operator()() __qpu__ {
    cudaq::qubit qb;
    bool b = (bool)(true);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(true);
    if (u8 == 1) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(true);
    if (i8 == 1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(true);
    if (u16 == 1) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(true);
    if (i16 == 1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(true);
    if (u32 == 1) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(true);
    if (i32 == 1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(true);
    if (u64 == 1) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(true);
    if (i64 == 1) {
      x(qi64);
    }
  }
};

struct testInt8 {
  auto operator()() __qpu__ {
    cudaq::qubit qb;
    bool b = (bool)((std::int8_t)-1);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)((std::int8_t)-1);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)((std::int8_t)-1);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)((std::int8_t)-1);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)((std::int8_t)-1);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)((std::int8_t)-1);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)((std::int8_t)-1);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)((std::int8_t)-1);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)((std::int8_t)-1);
    if (i64 == -1) {
      x(qi64);
    }
  }
};

struct testInt16 {
  auto operator()() __qpu__ {
    cudaq::qubit qb;
    bool b = (bool)((std::int16_t)-1);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)((std::int16_t)-1);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)((std::int16_t)-1);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)((std::int16_t)-1);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)((std::int16_t)-1);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)((std::int16_t)-1);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)((std::int16_t)-1);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)((std::int16_t)-1);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)((std::int16_t)-1);
    if (i64 == -1) {
      x(qi64);
    }
  }
};

struct testInt32 {
  auto operator()() __qpu__ {
    cudaq::qubit qb;
    bool b = (bool)((std::int32_t)-1);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)((std::int32_t)-1);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)((std::int32_t)-1);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)((std::int32_t)-1);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)((std::int32_t)-1);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)((std::int32_t)-1);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)((std::int32_t)-1);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)((std::int32_t)-1);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)((std::int32_t)-1);
    if (i64 == -1) {
      x(qi64);
    }
  }
};

struct testInt64 {
  auto operator()() __qpu__ {
    cudaq::qubit qb;
    bool b = (bool)((std::int64_t)-1);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)((std::int64_t)-1);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)((std::int64_t)-1);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)((std::int64_t)-1);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)((std::int64_t)-1);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)((std::int64_t)-1);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)((std::int64_t)-1);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)((std::int64_t)-1);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)((std::int64_t)-1);
    if (i64 == -1) {
      x(qi64);
    }
  }
};

void printCounts(cudaq::sample_result &result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << std::endl;
  }
}

int main() {
  {
    std::string expected = "";
    bool b = (bool)(true);
    expected += b == true ? "1" : "0";

    uint8_t u8 = (std::uint8_t)(true);
    expected += u8 == 1 ? "1" : "0";
    std::int8_t i8 = (std::int8_t)(true);
    expected += i8 == 1 ? "1" : "0";

    uint16_t u16 = (std::uint16_t)(true);
    expected += u16 == 1 ? "1" : "0";
    std::int16_t i16 = (std::int16_t)(true);
    expected += i16 == 1 ? "1" : "0";

    std::uint32_t u32 = (std::uint32_t)(true);
    expected += u32 == 1 ? "1" : "0";
    std::int32_t i32 = (std::int32_t)(true);
    expected += i32 == 1 ? "1" : "0";

    std::uint64_t u64 = (std::uint64_t)(true);
    expected += u64 == 1 ? "1" : "0";
    std::int64_t i64 = (std::int64_t)(true);
    expected += i64 == 1 ? "1" : "0";

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testBool{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    std::string expected = "";

    bool b = (bool)((std::int8_t)-1);
    expected += b == true ? "1" : "0";

    std::uint8_t u8 = (std::uint8_t)((std::int8_t)-1);
    expected += u8 == 255 ? "1" : "0";
    std::int8_t i8 = (std::int8_t)((std::int8_t)-1);
    expected += i8 == -1 ? "1" : "0";

    std::uint16_t u16 = (std::uint16_t)((std::int8_t)-1);
    expected += u16 == 65535 ? "1" : "0";
    std::int16_t i16 = (std::int16_t)((std::int8_t)-1);
    expected += i16 == -1 ? "1" : "0";

    std::uint32_t u32 = (std::uint32_t)((std::int8_t)-1);
    expected += u32 == 4294967295 ? "1" : "0";
    std::int32_t i32 = (std::int32_t)((std::int8_t)-1);
    expected += i32 == -1 ? "1" : "0";

    std::uint64_t u64 = (std::uint64_t)((std::int8_t)-1);
    expected += u64 == 18446744073709551615UL ? "1" : "0";
    std::int64_t i64 = (std::int64_t)((std::int8_t)-1);
    expected += i64 == -1 ? "1" : "0";

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt8{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    std::string expected = "";

    bool b = (bool)((std::int16_t)-1);
    expected += b == true ? "1" : "0";

    std::uint8_t u16_8 = (std::uint8_t)((std::int16_t)-1);
    expected += u16_8 == 255 ? "1" : "0";
    std::int8_t i16_8 = (std::int8_t)((std::int16_t)-1);
    expected += i16_8 == -1 ? "1" : "0";

    std::uint16_t u16 = (std::uint16_t)((std::int16_t)-1);
    expected += u16 == 65535U ? "1" : "0";
    std::int16_t i16 = (std::int16_t)((std::int16_t)-1);
    expected += i16 == -1 ? "1" : "0";

    std::uint32_t u32 = (std::uint32_t)((std::int16_t)-1);
    expected += u32 == 4294967295 ? "1" : "0";
    std::int32_t i32 = (std::int32_t)((std::int16_t)-1);
    expected += i32 == -1 ? "1" : "0";

    std::uint64_t u64 = (std::uint64_t)((std::int16_t)-1);
    expected += u64 == 18446744073709551615UL ? "1" : "0";
    std::int64_t i64 = (std::int64_t)((std::int16_t)-1);
    expected += i64 == -1 ? "1" : "0";

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt16{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    std::string expected = "";

    bool b = (bool)((std::int32_t)-1);
    expected += b == true ? "1" : "0";

    std::uint8_t u32_8 = (std::uint8_t)((std::int32_t)-1);
    expected += u32_8 == 255 ? "1" : "0";
    std::int8_t i32_8 = (std::int8_t)((std::int32_t)-1);
    expected += i32_8 == -1 ? "1" : "0";

    std::uint16_t u32_16 = (std::uint16_t)((std::int32_t)-1);
    expected += u32_16 == 65535U ? "1" : "0";
    std::int16_t i32_16 = (std::int16_t)((std::int32_t)-1);
    expected += i32_16 == -1 ? "1" : "0";

    std::uint32_t u32 = (std::uint32_t)((std::int32_t)-1);
    expected += u32 == 4294967295 ? "1" : "0";
    std::int32_t i32 = (std::int32_t)((std::int32_t)-1);
    expected += i32 == -1 ? "1" : "0";

    std::uint64_t u64 = (std::uint64_t)((std::int32_t)-1);
    expected += u64 == 18446744073709551615UL ? "1" : "0";
    std::int64_t i64 = (std::int64_t)((std::int32_t)-1);
    expected += i64 == -1 ? "1" : "0";

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt32{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    std::string expected = "";

    bool b = (bool)((std::int64_t)-1);
    expected += b == true ? "1" : "0";

    std::uint8_t u32_8 = (std::uint8_t)((std::int64_t)-1);
    expected += u32_8 == 255 ? "1" : "0";
    std::int8_t i32_8 = (std::int8_t)((std::int64_t)-1);
    expected += i32_8 == -1 ? "1" : "0";

    std::uint16_t u32_16 = (std::uint16_t)((std::int64_t)-1);
    expected += u32_16 == 65535U ? "1" : "0";
    std::int16_t i32_16 = (std::int16_t)((std::int64_t)-1);
    expected += i32_16 == -1 ? "1" : "0";

    std::uint32_t u32 = (std::uint32_t)((std::int64_t)-1);
    expected += u32 == 4294967295 ? "1" : "0";
    std::int32_t i32 = (std::int32_t)((std::int64_t)-1);
    expected += i32 == -1 ? "1" : "0";

    std::uint64_t u64 = (std::uint64_t)((std::int64_t)-1);
    expected += u64 == 18446744073709551615UL ? "1" : "0";
    std::int64_t i64 = (std::int64_t)((std::int64_t)-1);
    expected += i64 == -1 ? "1" : "0";

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt64{});
    printf("Actual:   ");
    printCounts(counts);
  }

  return 0;
}

// CHECK: Expected: 111111111
// CHECK: Actual:   111111111
// CHECK: Expected: 111111111
// CHECK: Actual:   111111111
// CHECK: Expected: 111111111
// CHECK: Actual:   111111111
// CHECK: Expected: 111111111
// CHECK: Actual:   111111111
// CHECK: Expected: 111111111
// CHECK: Actual:   111111111
