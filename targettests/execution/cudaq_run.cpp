/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ -fenable-cudaq-run %cpp_std %s -o %t && %t | FileCheck %s
// RUN: nvq++ -fenable-cudaq-run --library-mode %cpp_std %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ int nullary_test() {
  unsigned result = 0;
  cudaq::qvector v(8);
  h(v);
  z(v);
  for (int i = 0; i < 8; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (8 - 1 - i);
  }
  return result;
}

__qpu__ int unary_test(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v);
  z(v);
  for (int i = 0; i < count; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (count - 1 - i);
  }
  return result;
}

__qpu__ std::vector<bool> vector_bool_test() {
  std::vector<bool> vec{true, false};
  return vec;
}

__qpu__ std::vector<int> vector_int_test() {
  std::vector<int> result(2);
  result[0] = 42;
  result[1] = -13;
  return result;
}

__qpu__ std::vector<float> vector_float_test() {
  std::vector<float> result(3);
  result[0] = 3.141592653589;
  result[1] = 2.718281828459;
  result[2] = 6.62607015;
  return result;
}

/// TODO: this currently fails due to a missing support for tuple copy
// constructor in ConvertExpr.cpp
// __qpu__ std::tuple<int, bool> tuple_test(std::tuple<int, bool> tup) {
//   cudaq::qvector v(2);
//   return tup;
// }

struct MyTuple {
  bool boolVal;
  std::int64_t i64Val;
  double f64Val;
};

auto struct_test = []() __qpu__ {
  MyTuple t = {true, 654, 9.123};
  return t;
};

int main() {
  int c = 0;
  {
    const auto results = cudaq::run(100, nullary_test);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    const auto results = cudaq::run(50, unary_test, 4);
    c = 0;
    if (results.size() != 50) {
      printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  // TODO: this currently fails due to a missing support for tuple copy
  // constructor in ConvertExpr.cpp
  // {
  //   std::tuple<int, bool> t{13, true};
  //   const auto results = cudaq::run(50, unary_test_tuple, t);
  //   int c = 0;
  //   if (results.size() != 50) {
  //     printf("unary_test_list FAILED! Expected 50 shots. Got %lu\n",
  //     results.size());
  //   } else {
  //     printf("Got %lu\n", results.size());
  //     for (auto i : results) {
  //       printf("%d: [", c++);
  //       auto [a,b] = i;
  //       printf("%d, %d", a, b);
  //       printf("]\n");
  //     }
  //     printf("success!\n");
  //   }
  // }

  // Run async
  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, 100, nullary_test).get();
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, 50, unary_test, 4).get();
    c = 0;
    if (results.size() != 50) {
      printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

#if 0
  // vector return types are not fully supported yet.
  {
    const std::vector<std::vector<bool>> results =
        cudaq::run(3, vector_bool_test);
    c = 0;
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, (bool)i[0], (bool)i[1]);
        assert(i[0] == true);
        assert(i[1] == false);
      }
      printf("success!\n");
    }
  }

  {
    const std::vector<std::vector<int>> results =
        cudaq::run(3, vector_int_test);
    c = 0;
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, i[0], i[1]);
        assert(i[0] == 42);
        assert(i[1] == -13);
      }
      printf("success!\n");
    }
  }

  {
    const std::vector<std::vector<float>> results =
        cudaq::run(2, vector_float_test);
    c = 0;
    if (results.size() != 2) {
      printf("FAILED! Expected 2 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: {%f , %f , %f}\n", c++, i[0], i[1], i[2]);
      printf("success!\n");
    }
  }
#endif

  {
    const auto results = cudaq::run(3, struct_test);
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: {%s, %ld, %f}\n", c++, i.boolVal ? "true" : "false",
               i.i64Val, i.f64Val);
      printf("success!\n");
    }
  }

  return 0;
}

// CHECK: success!
// CHECK: success!
// CHECK: success!
// CHECK: success!
// XXECK: success!
// XXECK: success!
// XXECK: success!
// CHECK: success!
