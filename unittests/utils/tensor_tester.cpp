/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/utils/tensor.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <tuple>

TEST(CoreTester, checkTensorSimple) {
  auto registeredNames = cudaq::details::tensor_impl<>::get_registered();
  EXPECT_EQ(registeredNames.size(), 1);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "xtensorcomplex<double>") != registeredNames.end());

  {
    cudaq::tensor t({1, 2, 1});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}).real(), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2.2;
    EXPECT_NEAR(t.at({0, 1, 0}).real(), 2.2, 1e-8);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }

  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.copy(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.borrow(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaq::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.borrow(data.data()); }, std::runtime_error);
  }
  {
    cudaq::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.copy(data.data()); }, std::runtime_error);
  }
  {
    cudaq::tensor t;
    const std::vector<std::complex<double>> idata{1, 2, 3, 4};
    auto data = std::make_unique<std::complex<double>[]>(4);
    std::copy(idata.begin(), idata.end(), data.get());
    EXPECT_THROW({ t.take(data); }, std::runtime_error);
  }
  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.copy(data.data(), {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.copy(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.borrow(data.data(), {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }

  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.borrow(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaq::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.borrow(data.data()); }, std::runtime_error);
  }
  {
    cudaq::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.copy(data.data()); }, std::runtime_error);
  }
  {
    cudaq::tensor t;
    const std::vector<std::complex<double>> idata{1, 2, 3, 4};
    auto data = std::make_unique<std::complex<double>[]>(4);
    std::copy(idata.begin(), idata.end(), data.get());
    EXPECT_THROW({ t.take(data); }, std::runtime_error);
  }
  {
    cudaq::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    auto data = std::make_unique<std::complex<double>[]>(4);
    double count = 1.0;
    std::generate_n(data.get(), 4, [&]() { return count++; });
    t.take(data, {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }

  {
    cudaq::tensor<int> t({1, 2, 1});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2;
    EXPECT_EQ(t.at({0, 1, 0}), 2);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }
}

TEST(TensorTest, ConstructorWithShape) {
  std::vector<std::size_t> shape = {2, 3, 4};
  cudaq::tensor t(shape);

  EXPECT_EQ(t.rank(), 3);
  EXPECT_EQ(t.size(), 24);
  EXPECT_EQ(t.shape(), shape);
}

TEST(TensorTest, ConstructorWithDataAndShape) {
  std::vector<std::size_t> shape = {2, 2};
  std::complex<double> *data = new std::complex<double>[4];
  data[0] = {1.0, 0.0};
  data[1] = {0.0, 1.0};
  data[2] = {0.0, -1.0};
  data[3] = {1.0, 0.0};

  cudaq::tensor t(data, shape);

  EXPECT_EQ(t.rank(), 2);
  EXPECT_EQ(t.size(), 4);
  EXPECT_EQ(t.shape(), shape);

  // Check if data is correctly stored
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, AccessElements) {
  std::vector<std::size_t> shape = {2, 3};
  cudaq::tensor t(shape);

  // Set values
  t.at({0, 0}) = {1.0, 0.0};
  t.at({0, 1}) = {0.0, 1.0};
  t.at({1, 2}) = {-1.0, 0.0};

  // Check values
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 2}), std::complex<double>(-1.0, 0.0));
}

TEST(TensorTest, CopyData) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaq::tensor t(shape);

  t.copy(data.data(), shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, TakeData) {
  std::vector<std::size_t> shape = {2, 2};
  auto data = std::make_unique<std::complex<double>[]>(4);
  const std::vector<std::complex<double>> idata{
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  std::copy(idata.begin(), idata.end(), data.get());
  cudaq::tensor t(shape);

  t.take(data, shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));

  // Note: We don't delete data here as the tensor now owns it
}

TEST(TensorTest, BorrowData) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaq::tensor t(shape);

  t.borrow(data.data(), shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, InvalidAccess) {
  std::vector<std::size_t> shape = {2, 2};
  cudaq::tensor t(shape);

  try {
    t.at({2, 0});
  } catch (std::exception &e) {
    std::cout << "Exception: (" << typeid(e).name() << ")" << e.what()
              << std::endl;
  }
  EXPECT_THROW(t.at({2, 0}), std::runtime_error);
  EXPECT_THROW(t.at({0, 2}), std::runtime_error);
  EXPECT_THROW(t.at({0, 0, 0}), std::runtime_error);
}

TEST(TensorTest, checkNullaryConstructor) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaq::tensor t;

  t.copy(data.data(), shape);
}
