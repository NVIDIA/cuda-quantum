/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include "gtest/gtest.h"

TEST(Tensor, initialization) {
  {
    cudaq::matrix_2 m0;
    EXPECT_EQ(m0.dump(), "{}");
  }
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});
    EXPECT_EQ(m1.dump(), "{  { (1,0)  (0,0) }\n   { (0,0)  (1,0) }\n }");
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});
    EXPECT_EQ(m1.dump(),
              "{  { (1,0)  (2,0)  (3,0) }\n   { (4,0)  (5,0)  (6,0) }\n }");
  }
}

TEST(Tensor, initializationError) {
  {
    EXPECT_THROW(cudaq::matrix_2 m1({1., 2., 3., 4., 5.}, {2, 3}),
                 std::runtime_error);
  }
}

TEST(Tensor, access) {
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});

    EXPECT_EQ((m1[{0, 1}]), 0.);

    m1[{0, 1}] = 4.;
    m1[{1, 0}] = m1[{0, 1}];

    EXPECT_EQ((m1[{0, 1}]), 4.);
    EXPECT_EQ((m1[{1, 0}]), 4.);
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_EQ((m1[{0, 2}]), 3.);

    m1[{0, 2}] = 9.;
    m1[{1, 0}] = m1[{0, 2}];

    EXPECT_EQ((m1[{0, 2}]), 9.);
    EXPECT_EQ((m1[{1, 0}]), 9.);
  }
}

TEST(Tensor, accessError) {
  {
    cudaq::matrix_2 m0;

    EXPECT_THROW((m0[{0}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 1}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::matrix_2 m1({1., 0., 0., 1.});

    EXPECT_THROW((m1[{0, 2}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::matrix_2 m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_THROW((m1[{0, 3}]), std::runtime_error);
    EXPECT_THROW((m1[{2, 1}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 2, 3}]), std::runtime_error);
  }
}

TEST(Tensor, product) {
  {
    cudaq::matrix_2 m2({2., 1., 3., 4.});
    cudaq::matrix_2 m3({3., 2., 1., 4.});
    cudaq::matrix_2 m4 = m2 * m3;
    EXPECT_EQ(m4.dump(), "{  { (7,0)  (8,0) }\n   { (13,0)  (22,0) }\n }");
  }
  {
    cudaq::matrix_2 m2({1., 2., 3., 4., 5., 6.}, {3, 2});
    cudaq::matrix_2 m3({1., 2., 3., 4., 5., 6.}, {2, 3});
    cudaq::matrix_2 m4 = m2 * m3;
    EXPECT_EQ(m4.dump(), "{  { (9,0)  (12,0) }\n   { (15,0)  (19,0) }\n   { "
                         "(26,0)  (33,0) }\n }");
  }
}

TEST(Tensor, productError) {
  {
    cudaq::matrix_2 m2({2., 1., 3., 4.});
    cudaq::matrix_2 m3({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m2 * m3, std::runtime_error);
  }
}

TEST(Tensor, addition) {
  {
    cudaq::matrix_2 m5({2., 11., 3., 4.2});
    cudaq::matrix_2 m6({3., 42., 1.4, 4.});
    cudaq::matrix_2 m7 = m5 + m6;
    EXPECT_EQ(m7.dump(), "{  { (5,0)  (53,0) }\n   { (4.4,0)  (8.2,0) }\n }");
  }
}

TEST(Tensor, additionError) {
  {
    cudaq::matrix_2 m5({2., 1., 3., 4.});
    cudaq::matrix_2 m6({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m5 + m6, std::runtime_error);
  }
}

TEST(Tensor, subtraction) {
  {
    cudaq::matrix_2 m8({12.1, 1., 3., 14.});
    cudaq::matrix_2 m9({3., 22., 31., 4.});
    cudaq::matrix_2 ma = m8 - m9;
    EXPECT_EQ(ma.dump(), "{  { (9.1,0)  (-21,0) }\n   { (-28,0)  (10,0) }\n }");
  }
}
TEST(Tensor, subtractionError) {
  {
    cudaq::matrix_2 m8({2., 1., 3., 4.});
    cudaq::matrix_2 m9({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m8 - m9, std::runtime_error);
  }
}

TEST(Tensor, kroneckerProduct) {
  {
    cudaq::matrix_2 mb({6.1, 1.5, 3., 14.});
    cudaq::matrix_2 mc({7.4, 8., 9., 4.2});
    cudaq::matrix_2 md = cudaq::kronecker(mb, mc);
    EXPECT_EQ(
        md.dump(),
        "{  { (45.14,0)  (48.8,0)  (11.1,0)  (12,0) }\n   { (54.9,0)  "
        "(25.62,0)  (13.5,0)  (6.3,0) }\n   { (22.2,0)  (24,0)  (103.6,0)  "
        "(112,0) }\n   { (27,0)  (12.6,0)  (126,0)  (58.8,0) }\n }");
  }
}

TEST(Tensor, kroneckerOnList) {
  {
    cudaq::matrix_2 me({{1., 1.}}, {1, 1});
    cudaq::matrix_2 mf({1., 2.}, {1, 2});
    cudaq::matrix_2 mg({3., 4., 5.}, {3, 1});
    std::vector<cudaq::matrix_2> v{me, mf, mg};
    cudaq::matrix_2 mh = cudaq::kronecker(v.begin(), v.end());
    EXPECT_EQ(
        mh.dump(),
        "{  { (3,3)  (6,6) }\n   { (4,4)  (8,8) }\n   { (5,5)  (10,10) }\n }");
  }
}
