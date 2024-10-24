/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <tuple>
#include <vector>

void cuda_tensor() {
  {
    // Check if constructor compiles
    std::vector<std::size_t> shape = {2, 2};
    float *data = new float[4];
    cudaq::tensor<float> t(data, shape);

    // Check if `rank`, `size`, and `shape`  compile
    assert(t.rank() == 2);
    assert(t.size() == 4);
    assert(t.shape() == shape);

    // Check if data access compiles
    assert(t.at({0, 0}) == 1.0);
    assert(t.at({0, 1}) == 0.0);
    assert(t.at({1, 0}) == -1.0);
    assert(t.at({1, 1}) == 2.0);
  }

  {
    // Check if constructor compiles
    std::vector<std::size_t> shape = {2, 2};
    std::complex<double> *data = new std::complex<double>[4];
    cudaq::tensor t(data, shape);

    // Check if `rank`, `size`, and `shape`  compile
    assert(t.rank() == 2);
    assert(t.size() == 4);
    assert(t.shape() == shape);

    // Check if data access compiles
    assert(t.at({0, 0}) == std::complex<double>(0.0, 0.0));
    assert(t.at({0, 1}) == std::complex<double>(0.0, 0.0));
    assert(t.at({1, 0}) == std::complex<double>(0.0, 0.0));
    assert(t.at({1, 1}) == std::complex<double>(0.0, 0.0));
  }

  {
    // Check if constructor compiles
    cudaq::tensor t({2, 2});
    std::vector<std::complex<double>> data{1, 2, 3, 4};

    // Check if `copy` compiles
    t.copy(data.data(), {2, 2});
  }
  {
    // Check if constructor compiles
    cudaq::tensor t({2, 2});
    std::vector<std::complex<double>> data{1, 2, 3, 4};

    // Check if `copy` compiles
    t.copy(data.data());
  }

  {
    cudaq::tensor t({2, 2});
    std::vector<std::complex<double>> data{1, 2, 3, 4};

    // Check if `borrow` compiles
    t.borrow(data.data());
  }
  {
    cudaq::tensor t({2, 2});
    std::vector<std::complex<double>> data{1, 2, 3, 4};

    // Check if `borrow` compiles
    t.borrow(data.data(), {2, 2});
  }

  {
    cudaq::tensor t({2, 2});
    auto data = std::make_unique<std::complex<double>[]>(4);
    double count = 1.0;
    std::generate_n(data.get(), 4, [&]() { return count++; });

    // Check if take compiles
    t.take(data);
  }

  {
    cudaq::tensor t({2, 2});
    auto data = std::make_unique<std::complex<double>[]>(4);
    double count = 1.0;
    std::generate_n(data.get(), 4, [&]() { return count++; });

    // Check if take compiles
    t.take(data, {2, 2});
  }

  {
    cudaq::tensor<double> t({2, 2});

    // Check if `dump` compiles
    t.dump();
    // Check of `data` compiles
    assert(!t.data());
  }
}