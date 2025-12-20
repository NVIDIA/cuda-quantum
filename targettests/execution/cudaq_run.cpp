/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s
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

__qpu__ std::vector<bool> vector_bool_dynamic_size_test(int size) {
  std::vector<bool> vec(size);
  for (int i = 0; i < size; i++)
    vec[i] = (i % 2 == 0) ? true : false;
  return vec;
}

__qpu__ bool mz_test() {
  cudaq::qubit q;
  h(q);
  return mz(q);
}

struct vector_mz_test {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(5);
    cudaq::qubit p;
    x(q);
#ifdef CUDAQ_LIBRARY_MODE
    return cudaq::measure_result::to_bool_vector(mz(q));
#else
    return mz(q);
#endif
  }
};

struct vector_mz_dynamic_size_test {
  std::vector<bool> operator()(int size) __qpu__ {
    cudaq::qvector q(size);
    cudaq::qubit p;
    x(q);
#ifdef CUDAQ_LIBRARY_MODE
    return cudaq::measure_result::to_bool_vector(mz(q));
#else
    return mz(q);
#endif
  }
};

__qpu__ std::vector<int> vector_int_test() {
  std::vector<int> result(2);
  result[0] = 42;
  result[1] = -13;
  return result;
}

__qpu__ std::vector<int> vector_int_dynamic_size_test(int size) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    if (i == 0)
      result[i] = 42;
    else if (i == 1)
      result[i] = -13;
    else
      result[i] = 0;
  }
  return result;
}

__qpu__ std::vector<float> vector_float_test() {
  std::vector<float> result(3);
  result[0] = 3.141592653589;
  result[1] = 2.718281828459;
  result[2] = 6.62607015;
  return result;
}

__qpu__ std::vector<float> vector_float_dynamic_size_test(int size) {
  std::vector<float> result(size);
  for (int i = 0; i < size; i++) {
    if (i == 0)
      result[i] = 3.141592653589;
    else if (i == 1)
      result[i] = 2.718281828459;
    else if (i == 2)
      result[i] = 6.62607015;
    else
      result[i] = 0.0;
  }
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
  std::size_t shots = 10;
  int c = 0;
  {
    const auto results = cudaq::run(shots, nullary_test);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success - nullary_test\n");
    }
  }

  {
    const auto results = cudaq::run(shots, unary_test, 4);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success - unary_test\n");
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
        cudaq::run_async(/*qpu_id=*/0, shots, nullary_test).get();
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success async - nullary_test\n");
    }
  }

  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, shots, unary_test, 4).get();
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success async - unary_test\n");
    }
  }

  shots = 5;
  {
    const std::vector<std::vector<bool>> results =
        cudaq::run(shots, vector_bool_test);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, (bool)i[0], (bool)i[1]);
        assert(i[0] == true);
        assert(i[1] == false);
      }
      printf("success - vector_bool_test\n");
    }
  }

  {
    const std::vector<std::vector<bool>> results =
        cudaq::run(shots, vector_bool_dynamic_size_test, 2);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, (bool)i[0], (bool)i[1]);
        assert(i[0] == true);
        assert(i[1] == false);
      }
      printf("success - vector_bool_dynamic_size_test\n");
    }
  }

  {
    const std::vector<bool> results = cudaq::run(shots, mz_test);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, (bool)i);
      printf("success - mz_test\n");
    }
  }

  {
    const std::vector<std::vector<bool>> results =
        cudaq::run(shots, vector_mz_test{});
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {", c++);
        for (auto b : i)
          printf("%d ", (bool)b);
        printf("}\n");
      }
      printf("success - vector_mz_test\n");
    }
  }

  {
    const std::vector<std::vector<bool>> results =
        cudaq::run(shots, vector_mz_dynamic_size_test{}, 5);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {", c++);
        for (auto b : i)
          printf("%d ", (bool)b);
        printf("}\n");
      }
      printf("success - vector_mz_dynamic_size_test\n");
    }
  }

  {
    const std::vector<std::vector<int>> results =
        cudaq::run(shots, vector_int_test);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, i[0], i[1]);
        assert(i[0] == 42);
        assert(i[1] == -13);
      }
      printf("success - vector_int_test\n");
    }
  }

  {
    const std::vector<std::vector<int>> results =
        cudaq::run(shots, vector_int_dynamic_size_test, 2);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: {%d , %d}\n", c++, i[0], i[1]);
        assert(i[0] == 42);
        assert(i[1] == -13);
      }
      printf("success - vector_int_dynamic_size_test\n");
    }
  }

  {
    const std::vector<std::vector<float>> results =
        cudaq::run(shots, vector_float_test);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: {%f , %f , %f}\n", c++, i[0], i[1], i[2]);
      printf("success - vector_float_test\n");
    }
  }

  {
    const std::vector<std::vector<float>> results =
        cudaq::run(shots, vector_float_dynamic_size_test, 3);
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results)
        printf("%d: {%f , %f , %f}\n", c++, i[0], i[1], i[2]);
      printf("success - vector_float_dynamic_size_test\n");
    }
  }

  {
    const auto results = cudaq::run(shots, struct_test);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: {%s, %ld, %f}\n", c++, i.boolVal ? "true" : "false",
               i.i64Val, i.f64Val);
      printf("success - struct_test\n");
    }
  }

  return 0;
}

// CHECK: success - nullary_test
// CHECK: success - unary_test
// CHECK: success async - nullary_test
// CHECK: success async - unary_test
// CHECK: success - vector_bool_test
// CHECK: success - vector_bool_dynamic_size_test
// CHECK: success - mz_test
// CHECK: success - vector_mz_test
// CHECK: success - vector_mz_dynamic_size_test
// CHECK: success - vector_int_test
// CHECK: success - vector_int_dynamic_size_test
// CHECK: success - vector_float_test
// CHECK: success - vector_float_dynamic_size_test
// CHECK: success - struct_test
