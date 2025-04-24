/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -fenable-cudaq-run %cpp_std %s -o %t && %t | FileCheck %s

#include <cudaq.h>

// __qpu__ int nullary_test() {
//   unsigned result = 0;
//   cudaq::qvector v(8);
//   h(v);
//   z(v);
//   for (int i = 0; i < 8; i++) {
//     bool w = mz(v[i]);
//     result |= ((unsigned)w) << (8 - 1 - i);
//   }
//   return result;
// }

// __qpu__ int unary_test(int count) {
//   unsigned result = 0;
//   cudaq::qvector v(count);
//   h(v);
//   z(v);
//   for (int i = 0; i < count; i++) {
//     bool w = mz(v[i]);
//     result |= ((unsigned)w) << (count - 1 - i);
//   }
//   return result;
// }

// __qpu__ std::vector<float> unary_test_list_int(int count) {
//   cudaq::qvector v(1);
//   std::vector<float> vec {0, 1};
//   return vec;
// }

__qpu__ int unary_test_tuple(std::tuple<int, bool> tup) {
  cudaq::qvector v(2);
  return std::get<1>(tup);
}


int main() {
  // int c = 0;
  // {
  //   const auto results = cudaq::run(100, nullary_test);
  //   if (results.size() != 100) {
  //     printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
  //   } else {
  //     for (auto i : results)
  //       printf("%d: %d\n", c++, i);
  //     printf("success!\n");
  //   }
  // }

  // {
  //   const auto results = cudaq::run(50, unary_test, 4);
  //   c = 0;
  //   if (results.size() != 50) {
  //     printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
  //   } else {
  //     for (auto i : results)
  //       printf("%d: %d\n", c++, i);
  //     printf("success!\n");
  //   }
  // }

  {
    // const auto results = cudaq::run(1, unary_test_list_int, 4);
    // printf("Got %lu\n", results.size());
    // int c = 0;
    // if (results.size() != 10) {
    //   printf("unary_test_list FAILED! Expected 50 shots. Got %lu\n", results.size());
    // } else {
    //   printf("Got %lu\n", results.size());
    //   for (auto i : results) {
    //     printf("%d: [", c++);
    //     for (auto j : i) 
    //       printf("%d,", j);
    //     printf("]\n");
    //   }
    //   printf("success!\n");
    // }
  }

  {
    std::tuple<int, bool> t{13, true};
    const auto results = cudaq::run(50, unary_test_tuple, t);
    int c = 0;
    if (results.size() != 50) {
      printf("unary_test_list FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      printf("Got %lu\n", results.size());
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        // printf("%d: [", c++);
        // auto [a,b] = i;
        // printf("%d, %d", a, b);
        // printf("]\n");
      }
      printf("success!\n");
    }
  }

  // // Run async
  // {
  //   const auto results =
  //       cudaq::run_async(/*qpu_id=*/0, 100, nullary_test).get();
  //   if (results.size() != 100) {
  //     printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
  //   } else {
  //     for (auto i : results)
  //       printf("%d: %d\n", c++, i);
  //     printf("success!\n");
  //   }
  // }

  // {
  //   const auto results =
  //       cudaq::run_async(/*qpu_id=*/0, 50, unary_test, 4).get();
  //   c = 0;
  //   if (results.size() != 50) {
  //     printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
  //   } else {
  //     for (auto i : results)
  //       printf("%d: %d\n", c++, i);
  //     printf("success!\n");
  //   }
  // }

  return 0;
}

// CHECK: success!
// CHECK: success!
// CHECK: success!
// CHECK: success!
