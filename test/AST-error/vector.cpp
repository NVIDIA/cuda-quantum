/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std -verify %s

#include <cudaq.h>

// We don't allow building vectors in kernels (yet).
struct VectorVectorReturner {
  // expected-error@*{{constructor within quantum kernel is not allowed}}
  std::vector<std::vector<double>>
  operator()(std::vector<std::vector<int>> theta) __qpu__ {
    std::vector<std::vector<double>> result;
    for (std::size_t i = 0, N = theta.size(); i < N; ++i) {
      auto &v = theta[i];
      auto &r = result[i];
      for (std::size_t j = 0, M = v.size(); j < M; ++j)
        r[j] = v[j];
    }
    // expected-error@+1{{C++ constructor (non-default)}}
    return result;
  }
};
