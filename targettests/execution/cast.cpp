/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

template <typename S, typename D>
std::string cast_and_compare(S val, D expected) {
  return ((D)val) == expected ? "1" : "0";
}

struct testBool {
  auto operator()() __qpu__ {
    cudaq::qubit qb4;
    bool b4 = (std::int32_t)(bool)(4);
    if (b4 == 1) {
      x(qb4);
    }

    cudaq::qubit qb_4;
    bool b_4 = (std::int32_t)(bool)(-2);
    if (b_4 == 1) {
      x(qb_4);
    }

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

    cudaq::qubit qf32;
    float f32 = (float)(true);
    if (f32 == 1.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(true);
    if (f64 == 1.0) {
      x(qf64);
    }
  }
};

struct testInt8 {
  auto operator()() __qpu__ {
    constexpr std::int8_t negOne = -1;

    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == -1.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(negOne);
    if (f64 == -1.0) {
      x(qf64);
    }
  }
};

struct testUInt8 {
  auto operator()() __qpu__ {
    constexpr std::uint8_t negOne = -1;

    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 255) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == 255) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 255) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == 255) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 255) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == 255) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == 255.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(negOne);
    if (f64 == 255.0) {
      x(qf64);
    }
  }
};

struct testInt16 {
  auto operator()() __qpu__ {
    constexpr std::int16_t negOne = -1;
    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == -1.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(negOne);
    if (f64 == -1.0) {
      x(qf64);
    }
  }
};

struct testUInt16 {
  auto operator()() __qpu__ {
    constexpr std::uint16_t negOne = -1;
    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 65535) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 65535) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == 65335.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(negOne);
    if (f64 == 65335.0) {
      x(qf64);
    }
  }
};

struct testInt32 {
  auto operator()() __qpu__ {
    constexpr std::int32_t negOne = -1;
    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == -1.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    std::int64_t f64 = (std::int64_t)(negOne);
    if (f64 == -1.0) {
      x(qf64);
    }
  }
};

struct testUInt32 {
  auto operator()() __qpu__ {
    constexpr std::uint32_t negOne = -1;
    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 4294967295) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == 4294967295.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    std::int64_t f64 = (std::int64_t)(negOne);
    if (f64 == 4294967295.0) {
      x(qf64);
    }
  }
};

struct testInt64 {
  auto operator()() __qpu__ {
    constexpr std::int64_t negOne = -1;
    cudaq::qubit qb;
    bool b = (bool)(negOne);
    if (b == true) {
      x(qb);
    }

    cudaq::qubit qu8;
    std::uint8_t u8 = (std::uint8_t)(negOne);
    if (u8 == 255) {
      x(qu8);
    }
    cudaq::qubit qi8;
    std::int8_t i8 = (std::int8_t)(negOne);
    if (i8 == -1) {
      x(qi8);
    }

    cudaq::qubit qu16;
    std::uint16_t u16 = (std::uint16_t)(negOne);
    if (u16 == 65535) {
      x(qu16);
    }
    cudaq::qubit qi16;
    std::int16_t i16 = (std::int16_t)(negOne);
    if (i16 == -1) {
      x(qi16);
    }

    cudaq::qubit qu32;
    std::uint32_t u32 = (std::uint32_t)(negOne);
    if (u32 == 4294967295) {
      x(qu32);
    }
    cudaq::qubit qi32;
    std::int32_t i32 = (std::int32_t)(negOne);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu64;
    std::uint64_t u64 = (std::uint64_t)(negOne);
    if (u64 == 18446744073709551615UL) {
      x(qu64);
    }
    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(negOne);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qf32;
    float f32 = (float)(negOne);
    if (f32 == -1.0f) {
      x(qf32);
    }
    cudaq::qubit qf64;
    double f64 = (double)(negOne);
    if (f64 == -1) {
      x(qf64);
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

struct testConsecutiveCasts {
  auto operator()() __qpu__ {
    cudaq::qubit qi32;
    std::int64_t i32 = (std::int32_t)(std::int64_t)((std::int16_t)-1);
    if (i32 == -1) {
      x(qi32);
    }

    cudaq::qubit qu32;
    std::int64_t u32 = (std::int32_t)(std::uint64_t)((std::int16_t)-1);
    if (u32 == -1) {
      x(qu32);
    }

    cudaq::qubit qi64;
    std::int64_t i64 = (std::int64_t)(std::int16_t)((std::int32_t)-1);
    if (i64 == -1) {
      x(qi64);
    }

    cudaq::qubit qu64;
    std::int64_t u64 = (std::int64_t)(std::uint16_t)((std::int32_t)-1);
    if (u64 == 65535) {
      x(qu64);
    }
  }
};

struct testUnsignedTruncation {
  auto operator()() __qpu__ {
    cudaq::qubit qi32;
    std::uint8_t i32 = (std::uint8_t)(std::uint32_t)(-1);
    if (i32 == 255) {
      x(qi32);
    }

    cudaq::qubit qi64;
    std::uint32_t i64 = (std::uint32_t)(std::uint64_t)(-1);
    if (i64 == 4294967295) {
      x(qi64);
    }
  }
};

struct testUnsignedExtension {
  auto operator()() __qpu__ {
    cudaq::qubit qi8;
    std::uint32_t i8 = (std::uint32_t)(std::uint8_t)(-1);
    if (i8 == 255) {
      x(qi8);
    }

    cudaq::qubit qi32;
    std::uint64_t i32 = (std::uint64_t)(std::uint32_t)(-1);
    if (i32 == 4294967295) {
      x(qi32);
    }
  }
};

struct testSignedToUnsignedTruncation {
  auto operator()() __qpu__ {
    cudaq::qubit qi32;
    std::uint8_t i32 = (std::uint8_t)(std::int32_t)(-1);
    if (i32 == 255) {
      x(qi32);
    }

    cudaq::qubit qi64;
    std::uint32_t i64 = (std::uint32_t)(std::int64_t)(-1);
    if (i64 == 4294967295) {
      x(qi64);
    }
  }
};

struct testSignedToUnsignedExtension {
  auto operator()() __qpu__ {
    cudaq::qubit qi8;
    std::uint32_t i8 = (std::uint32_t)(std::int8_t)(-1);
    if (i8 == 4294967295) {
      x(qi8);
    }

    cudaq::qubit qi32;
    std::uint64_t i32 = (std::uint64_t)(std::int32_t)(-1);
    if (i32 == 18446744073709551615UL) {
      x(qi32);
    }
  }
};

struct testUnsignedToSignedTruncation {
  auto operator()() __qpu__ {
    cudaq::qubit qi32;
    std::int8_t i32 = (std::int8_t)(std::uint32_t)(-1);
    if (i32 == (std::int8_t)(-1)) {
      x(qi32);
    }

    cudaq::qubit qi64;
    std::int32_t i64 = (std::int32_t)(std::uint64_t)(-1);
    if (i64 == std::int32_t(-1)) {
      x(qi64);
    }
  }
};

struct testUnsignedToSignedExtension {
  auto operator()() __qpu__ {
    cudaq::qubit qi8;
    std::int32_t i8 = (std::int32_t)(std::uint8_t)(-1);
    if (i8 == 255) {
      x(qi8);
    }

    cudaq::qubit qi32;
    std::int64_t i32 = (std::int64_t)(std::uint32_t)(-1);
    if (i32 == 4294967295) {
      x(qi32);
    }
  }
};

int main() {
  {
    printf("*** Test casts from bool ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int64_t, bool>(4, 4);
    expected += cast_and_compare<std::int64_t, bool>(-4, -4);

    expected += cast_and_compare<bool, bool>(true, 1);

    expected += cast_and_compare<bool, std::uint8_t>(true, 1);
    expected += cast_and_compare<bool, std::int8_t>(true, 1);

    expected += cast_and_compare<bool, std::uint16_t>(true, 1);
    expected += cast_and_compare<bool, std::int16_t>(true, 1);

    expected += cast_and_compare<bool, std::uint32_t>(true, 1);
    expected += cast_and_compare<bool, std::int32_t>(true, 1);

    expected += cast_and_compare<bool, std::uint64_t>(true, 1);
    expected += cast_and_compare<bool, std::int64_t>(true, 1);

    expected += cast_and_compare<bool, float>(true, 1.0f);
    expected += cast_and_compare<bool, double>(true, 1.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testBool{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from int8 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int8_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::int8_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::int8_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::int8_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::int8_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::int8_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::int8_t, std::int32_t>(-1, -1);

    expected += cast_and_compare<std::int8_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint64_t>::max());
    expected += cast_and_compare<std::int8_t, std::int64_t>(-1, -1);

    expected += cast_and_compare<std::int8_t, float>(-1, -1.0f);
    expected += cast_and_compare<std::int8_t, double>(-1, -1.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt8{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from uint8 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint8_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::uint8_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint8_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::uint8_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint8_t, std::int16_t>(
        -1, std::numeric_limits<std::uint8_t>::max());

    expected += cast_and_compare<std::uint8_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint8_t, std::int32_t>(
        -1, std::numeric_limits<std::uint8_t>::max());

    expected += cast_and_compare<std::uint8_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint8_t, std::int64_t>(
        -1, std::numeric_limits<std::uint8_t>::max());

    expected += cast_and_compare<std::uint8_t, float>(-1, 255.0f);
    expected += cast_and_compare<std::uint8_t, double>(-1, 255.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUInt8{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from int16 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int16_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::int16_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::int16_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::int16_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::int16_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::int16_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::int16_t, std::int32_t>(-1, -1);

    expected += cast_and_compare<std::int16_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint64_t>::max());
    expected += cast_and_compare<std::int16_t, std::int64_t>(-1, -1);

    expected += cast_and_compare<std::int16_t, float>(-1, -1.0f);
    expected += cast_and_compare<std::int16_t, double>(-1, -1.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt16{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from uint16 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint16_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::uint16_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint16_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::uint16_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::uint16_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::uint16_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::uint16_t, std::int32_t>(
        -1, std::numeric_limits<std::uint16_t>::max());

    expected += cast_and_compare<std::uint16_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::uint16_t, std::int64_t>(
        -1, std::numeric_limits<std::uint16_t>::max());

    expected += cast_and_compare<std::uint16_t, float>(-1, 65535.0f);
    expected += cast_and_compare<std::uint16_t, double>(-1, 65535.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUInt8{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from int32 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int32_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::int32_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::int32_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::int32_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::int32_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::int32_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::int32_t, std::int32_t>(-1, -1);

    expected += cast_and_compare<std::int32_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint64_t>::max());
    expected += cast_and_compare<std::int32_t, std::int64_t>(-1, -1);

    expected += cast_and_compare<std::int32_t, float>(-1, -1.0f);
    expected += cast_and_compare<std::int32_t, double>(-1, -1.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt32{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from uint32 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint32_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::uint32_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::uint32_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::uint32_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::uint32_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::uint32_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::uint32_t, std::int32_t>(-1, -1);

    expected += cast_and_compare<std::uint32_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::uint32_t, std::int64_t>(
        -1, std::numeric_limits<std::uint32_t>::max());

    expected += cast_and_compare<std::uint32_t, float>(-1, 4294967295.0f);
    expected += cast_and_compare<std::uint32_t, double>(-1, 4294967295.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUInt8{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test casts from int64 ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int64_t, bool>(
        -1, std::numeric_limits<bool>::max());

    expected += cast_and_compare<std::int64_t, std::uint8_t>(
        -1, std::numeric_limits<std::uint8_t>::max());
    expected += cast_and_compare<std::int64_t, std::int8_t>(-1, -1);

    expected += cast_and_compare<std::int64_t, std::uint16_t>(
        -1, std::numeric_limits<std::uint16_t>::max());
    expected += cast_and_compare<std::int64_t, std::int16_t>(-1, -1);

    expected += cast_and_compare<std::int64_t, std::uint32_t>(
        -1, std::numeric_limits<std::uint32_t>::max());
    expected += cast_and_compare<std::int64_t, std::int32_t>(-1, -1);

    expected += cast_and_compare<std::int64_t, std::uint64_t>(
        -1, std::numeric_limits<std::uint64_t>::max());
    expected += cast_and_compare<std::int64_t, std::int64_t>(-1, -1);

    expected += cast_and_compare<std::int64_t, float>(-1, -1.0f);
    expected += cast_and_compare<std::int64_t, double>(-1, -1.0);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testInt64{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test consecutive casts ***\n");
    std::string expected = "";
    expected +=
        cast_and_compare<std::int64_t, std::int32_t>((std::int16_t)-1, -1);
    expected +=
        cast_and_compare<std::uint64_t, std::int32_t>((std::int16_t)-1, -1);

    expected +=
        cast_and_compare<std::int16_t, std::int64_t>((std::int32_t)-1, -1);
    expected += cast_and_compare<std::uint16_t, std::int64_t>(
        (std::int32_t)-1, std::numeric_limits<std::uint16_t>::max());

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testConsecutiveCasts{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test unsigned truncation ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint32_t, std::uint8_t>(-1, 255);
    expected += cast_and_compare<std::uint64_t, std::uint32_t>(-1, 4294967295);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUnsignedTruncation{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test unsigned extension ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint8_t, std::uint32_t>(-1, 255);
    expected += cast_and_compare<std::uint32_t, std::uint64_t>(-1, 4294967295);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUnsignedExtension{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test signed to unsigned truncation ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int32_t, std::uint8_t>(-1, 255);
    expected += cast_and_compare<std::int64_t, std::uint32_t>(-1, 4294967295);

    printf("Expected: %s\n", expected.c_str());
    auto counts = cudaq::sample(testSignedToUnsignedTruncation{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test signed to unsigned extension ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int8_t, std::uint32_t>(-1, 4294967295);
    expected += cast_and_compare<std::int32_t, std::uint64_t>(
        -1, 18446744073709551615UL);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testSignedToUnsignedExtension{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test unsigned to signed truncation ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::int32_t, std::uint8_t>(-1, 255);
    expected += cast_and_compare<std::int64_t, std::uint32_t>(-1, 4294967295);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testSignedToUnsignedTruncation{});
    printf("Actual:   ");
    printCounts(counts);
  }

  {
    printf("*** Test unsigned to signed extension ***\n");
    std::string expected = "";
    expected += cast_and_compare<std::uint8_t, std::int32_t>(-1, 255);
    expected += cast_and_compare<std::uint32_t, std::int64_t>(-1, 4294967295);

    printf("Expected: %s\n", expected.c_str());

    auto counts = cudaq::sample(testUnsignedToSignedExtension{});
    printf("Actual:   ");
    printCounts(counts);
  }

  return 0;
}

// CHECK: *** Test casts from bool ***
// CHECK: Expected: 1111111111111
// CHECK: Actual:   1111111111111

// CHECK: *** Test casts from int8 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from uint8 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from int16 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from uint16 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from int32 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from uint32 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test casts from int64 ***
// CHECK: Expected: 11111111111
// CHECK: Actual:   11111111111

// CHECK: *** Test consecutive casts ***
// CHECK: Expected: 1111
// CHECK: Actual:   1111

// CHECK: *** Test unsigned truncation ***
// CHECK: Expected: 1
// CHECK: Actual:   1

// CHECK: *** Test unsigned extension ***
// CHECK: Expected: 1
// CHECK: Actual:   1

// CHECK: *** Test signed to unsigned truncation ***
// CHECK: Expected: 1
// CHECK: Actual:   1

// CHECK: *** Test signed to unsigned extension ***
// CHECK: Expected: 1
// CHECK: Actual:   1

// CHECK: *** Test unsigned to signed truncation ***
// CHECK: Expected: 1
// CHECK: Actual:   1

// CHECK: *** Test unsigned to signed  extension ***
// CHECK: Expected: 1
// CHECK: Actual:   1
