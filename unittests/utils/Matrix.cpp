/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/matrix.h"
#include "gtest/gtest.h"
#include <algorithm>

void string_equal(std::string str1, std::string str2) {
  auto is_space = [](char c) {
    return c == ' ';
  }; // *only* space, opposed to builtin isspace
  auto it = std::remove_if(str1.begin(), str1.end(), is_space);
  str1 = std::string(str1.begin(), it);
  it = std::remove_if(str2.begin(), str2.end(), is_space);
  str2 = std::string(str2.begin(), it);
  EXPECT_EQ(str1, str2);
}

TEST(Tensor, initialization) {
  {
    cudaq::complex_matrix m0;
    string_equal(m0.to_string(), "\n");
  }
  {
    cudaq::complex_matrix m1({1., 0., 0., 1.});
    string_equal(m1.to_string(), "(1,0) (0,0)\n(0,0) (1,0)\n");
  }
  {
    cudaq::complex_matrix m1({1., 2., 3., 4., 5., 6.}, {2, 3});
    string_equal(m1.to_string(), "(1,0) (2,0) (3,0)\n(4,0) (5,0) (6,0)\n");
  }
  {
    cudaq::complex_matrix m1({1., 2., 3., 4., 5., 6.}, {2, 3},
                             cudaq::complex_matrix::order::column_major);
    string_equal(m1.to_string(), "(1,0) (3,0) (5,0)\n(2,0) (4,0) (6,0)\n");
  }
}

TEST(Tensor, initializationError) {
  {
    EXPECT_THROW(cudaq::complex_matrix m1({1., 2., 3., 4., 5.}, {2, 3}),
                 std::runtime_error);
  }
}

TEST(Tensor, access) {
  {
    cudaq::complex_matrix m1({1., 0., 0., 1.});

    EXPECT_EQ((m1[{0, 1}]), 0.);

    m1[{0, 1}] = 4.;
    m1[{1, 0}] = m1[{0, 1}];

    EXPECT_EQ((m1[{0, 1}]), 4.);
    EXPECT_EQ((m1[{1, 0}]), 4.);
  }
  {
    cudaq::complex_matrix m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_EQ((m1[{0, 2}]), 3.);

    m1[{0, 2}] = 9.;
    m1[{1, 0}] = m1[{0, 2}];

    EXPECT_EQ((m1[{0, 2}]), 9.);
    EXPECT_EQ((m1[{1, 0}]), 9.);
  }

  {
    cudaq::complex_matrix m1({1., 2., 3., 4., 5., 6.}, {2, 3},
                             cudaq::complex_matrix::order::column_major);

    EXPECT_EQ((m1[{0, 2}]), 5.);

    m1[{0, 2}] = 9.;
    m1[{1, 0}] = m1[{0, 2}];

    EXPECT_EQ((m1[{0, 2}]), 9.);
    EXPECT_EQ((m1[{1, 0}]), 9.);
  }
}

TEST(Tensor, accessError) {
  {
    cudaq::complex_matrix m0;

    EXPECT_THROW((m0[{0}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 1}]), std::runtime_error);
    EXPECT_THROW((m0[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::complex_matrix m1({1., 0., 0., 1.});

    EXPECT_THROW((m1[{0, 2}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 0, 0}]), std::runtime_error);
  }
  {
    cudaq::complex_matrix m1({1., 2., 3., 4., 5., 6.}, {2, 3});

    EXPECT_THROW((m1[{0, 3}]), std::runtime_error);
    EXPECT_THROW((m1[{2, 1}]), std::runtime_error);
    EXPECT_THROW((m1[{0, 2, 3}]), std::runtime_error);
  }
}

TEST(Tensor, multiplication) {
  const auto vecToStr = [](const auto &vec) {
    auto it = vec.cbegin();
    if (it == vec.cend())
      return std::string("[]");

    std::stringstream ss;
    ss << "[(" << it->real() << ", " << it->imag() << ")";
    while (++it != vec.cend())
      ss << ", (" << it->real() << ", " << it->imag() << ")";
    ss << "]";
    return ss.str();
  };

  {
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix::value_type scalar(3.);
    string_equal((scalar * m2).to_string(), "(6,0) (3,0)\n(9,0) (12,0)\n");
  }
  {
    cudaq::complex_matrix m2({2., 3., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix::value_type scalar(3.);
    string_equal((scalar * m2).to_string(), "(6,0) (3,0)\n(9,0) (12,0)\n");
  }

  {
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    std::vector<cudaq::complex_matrix::value_type> vect = {5, 6};
    std::vector<cudaq::complex_matrix::value_type> expected = {16, 39};
    string_equal(vecToStr(m2 * vect), vecToStr(expected));
  }
  {
    cudaq::complex_matrix m2({2., 3., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    std::vector<cudaq::complex_matrix::value_type> vect = {5, 6};
    std::vector<cudaq::complex_matrix::value_type> expected = {16, 39};
    string_equal(vecToStr(m2 * vect), vecToStr(expected));
  }
}

TEST(Tensor, product) {
  {
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m3({3., 2., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m4 = m2 * m3;
    string_equal(m4.to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");
  }
  {
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m3({3., 1., 2., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m4 = m2 * m3;
    string_equal(m4.to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");
  }
  {
    cudaq::complex_matrix m2({2., 3., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m3({3., 2., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m4 = m2 * m3;
    string_equal(m4.to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");
  }
  {
    cudaq::complex_matrix m2({2., 3., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m3({3., 1., 2., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m4 = m2 * m3;
    string_equal(m4.to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");
  }

  {
    cudaq::complex_matrix m2({1., 2., 3., 4., 5., 6.}, {3, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m3({1., 2., 3., 4., 5., 6.}, {2, 3},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m4 = m2 * m3;
    cudaq::complex_matrix m5 = m3 * m2;
    string_equal(m4.to_string(), "(9,0) (12,0) (15,0) \n (19,0)"
                                 "(26,0) (33,0)\n (29, 0) (40, 0) (51, 0)\n");
    string_equal(m5.to_string(), "(22,0) (28,0)\n(49,0) (64,0)\n");
  }
  {
    cudaq::complex_matrix m2({1., 3., 5., 2., 4., 6.}, {3, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m3({1., 2., 3., 4., 5., 6.}, {2, 3},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m4 = m2 * m3;
    cudaq::complex_matrix m5 = m3 * m2;
    string_equal(m4.to_string(), "(9,0) (12,0) (15,0) \n (19,0)"
                                 "(26,0) (33,0)\n (29, 0) (40, 0) (51, 0)\n");
    string_equal(m5.to_string(), "(22,0) (28,0)\n(49,0) (64,0)\n");
  }
  {
    cudaq::complex_matrix m2({1., 2., 3., 4., 5., 6.}, {3, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m3({1., 4., 2., 5., 3., 6.}, {2, 3},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m4 = m2 * m3;
    cudaq::complex_matrix m5 = m3 * m2;
    string_equal(m4.to_string(), "(9,0) (12,0) (15,0) \n (19,0)"
                                 "(26,0) (33,0)\n (29, 0) (40, 0) (51, 0)\n");
    string_equal(m5.to_string(), "(22,0) (28,0)\n(49,0) (64,0)\n");
  }
  {
    cudaq::complex_matrix m2({1., 3., 5., 2., 4., 6.}, {3, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m3({1., 4., 2., 5., 3., 6.}, {2, 3},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m4 = m2 * m3;
    cudaq::complex_matrix m5 = m3 * m2;
    string_equal(m4.to_string(), "(9,0) (12,0) (15,0) \n (19,0)"
                                 "(26,0) (33,0)\n (29, 0) (40, 0) (51, 0)\n");
    string_equal(m5.to_string(), "(22,0) (28,0)\n(49,0) (64,0)\n");
  }
}

TEST(Tensor, productError) {
  {
    cudaq::complex_matrix m2({2., 1., 3., 4.});
    cudaq::complex_matrix m3({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m2 * m3, std::runtime_error);
  }
}

TEST(Tensor, addition) {
  {
    cudaq::complex_matrix m5({2., 11., 3., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m6({3., 42., 1.4, 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m7 = m5 + m6;
    string_equal(m7.to_string(), "(5,0) (53,0)\n(4.4,0) (8.2,0)\n");
  }
  {
    cudaq::complex_matrix m5({2., 3., 11., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m6({3., 42., 1.4, 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m7 = m5 + m6;
    string_equal(m7.to_string(), "(5,0) (53,0)\n(4.4,0) (8.2,0)\n");
  }
  {
    cudaq::complex_matrix m5({2., 11., 3., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m6({3., 1.4, 42., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m7 = m5 + m6;
    string_equal(m7.to_string(), "(5,0) (53,0)\n(4.4,0) (8.2,0)\n");
  }
  {
    cudaq::complex_matrix m5({2., 3., 11., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m6({3., 1.4, 42., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m7 = m5 + m6;
    string_equal(m7.to_string(), "(5,0) (53,0)\n(4.4,0) (8.2,0)\n");
  }
}

TEST(Tensor, additionError) {
  {
    cudaq::complex_matrix m5({2., 1., 3., 4.});
    cudaq::complex_matrix m6({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m5 + m6, std::runtime_error);
  }
}

TEST(Tensor, subtraction) {
  {
    cudaq::complex_matrix m8({12.1, 1., 3., 14.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m9({3., 22., 31., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix ma = m8 - m9;
    string_equal(ma.to_string(), "(9.1,0) (-21,0)\n(-28,0) (10,0)\n");
  }
  {
    cudaq::complex_matrix m8({12.1, 3., 1., 14.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m9({3., 22., 31., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix ma = m8 - m9;
    string_equal(ma.to_string(), "(9.1,0) (-21,0)\n(-28,0) (10,0)\n");
  }
  {
    cudaq::complex_matrix m8({12.1, 1., 3., 14.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m9({3., 31., 22., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix ma = m8 - m9;
    string_equal(ma.to_string(), "(9.1,0) (-21,0)\n(-28,0) (10,0)\n");
  }
  {
    cudaq::complex_matrix m8({12.1, 3., 1., 14.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m9({3., 31., 22., 4.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix ma = m8 - m9;
    string_equal(ma.to_string(), "(9.1,0) (-21,0)\n(-28,0) (10,0)\n");
  }
}
TEST(Tensor, subtractionError) {
  {
    cudaq::complex_matrix m8({2., 1., 3., 4.});
    cudaq::complex_matrix m9({1., 2., 3., 4., 5., 6.}, {3, 2});
    EXPECT_THROW(m8 - m9, std::runtime_error);
  }
}

TEST(Tensor, kroneckerProduct) {
  {
    cudaq::complex_matrix mb({6.1, 1.5, 3., 14.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix mc({7.4, 8., 9., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix md = cudaq::kronecker(mb, mc);
    string_equal(md.to_string(),
                 "(45.14,0) (48.8,0) (11.1,0) (12,0)\n(54.9,0) "
                 "(25.62,0) (13.5,0) (6.3,0)\n(22.2,0) (24,0) (103.6,0) "
                 "(112,0)\n(27,0) (12.6,0) (126,0) (58.8,0)\n");
  }
  {
    cudaq::complex_matrix mb({6.1, 3, 1.5, 14.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix mc({7.4, 8., 9., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix md = cudaq::kronecker(mb, mc);
    string_equal(md.to_string(),
                 "(45.14,0) (48.8,0) (11.1,0) (12,0)\n(54.9,0) "
                 "(25.62,0) (13.5,0) (6.3,0)\n(22.2,0) (24,0) (103.6,0) "
                 "(112,0)\n(27,0) (12.6,0) (126,0) (58.8,0)\n");
  }
  {
    cudaq::complex_matrix mb({6.1, 1.5, 3., 14.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix mc({7.4, 9., 8., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix md = cudaq::kronecker(mb, mc);
    string_equal(md.to_string(),
                 "(45.14,0) (48.8,0) (11.1,0) (12,0)\n(54.9,0) "
                 "(25.62,0) (13.5,0) (6.3,0)\n(22.2,0) (24,0) (103.6,0) "
                 "(112,0)\n(27,0) (12.6,0) (126,0) (58.8,0)\n");
  }
  {
    cudaq::complex_matrix mb({6.1, 3, 1.5, 14.}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix mc({7.4, 9., 8., 4.2}, {2, 2},
                             cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix md = cudaq::kronecker(mb, mc);
    string_equal(md.to_string(),
                 "(45.14,0) (48.8,0) (11.1,0) (12,0)\n(54.9,0) "
                 "(25.62,0) (13.5,0) (6.3,0)\n(22.2,0) (24,0) (103.6,0) "
                 "(112,0)\n(27,0) (12.6,0) (126,0) (58.8,0)\n");
  }
}

TEST(Tensor, kroneckerOnList) {
  {
    cudaq::complex_matrix me({{1., 1.}}, {1, 1});
    cudaq::complex_matrix mf({1., 2.}, {1, 2});
    cudaq::complex_matrix mg({3., 4., 5.}, {3, 1});
    std::vector<cudaq::complex_matrix> v{me, mf, mg};
    cudaq::complex_matrix mh = cudaq::kronecker(v.begin(), v.end());
    string_equal(mh.to_string(), "(3,3) (6,6)\n(4,4) (8,8)\n(5,5) (10,10)\n");
  }
}

TEST(Tensor, exponential) {
  {
    cudaq::complex_matrix me({1., 1., 0.5, 0.0}, {2, 2});
    cudaq::complex_matrix mf({1., 0., 1., .5, .7, 0., 1., 0., 2.}, {3, 3});
    cudaq::complex_matrix mg(
        {1., 0., .4, .6, .7, .8, .9, 0., .3, .1, .2, 1., 0., 0.5, 0.2, .5},
        {4, 4});

    auto me_exp = me.exponential();
    auto mf_exp = mf.exponential();
    auto mg_exp = mg.exponential();

    string_equal(me_exp.to_string(),
                 "(3.23795,0) (1.86268,0)\n(0.93134,0) (1.37527,0)\n");

    string_equal(mf_exp.to_string(),
                 "(4.84921,0) (0,0) (5.4755,0)\n(1.46673,0) (2.01375,0) "
                 "(0.977708,0)\n(5.4755,0) (0,0) (10.3247,0)\n");

    string_equal(mg_exp.to_string(),
                 "(2.9751,0) (0.447969,0) (1.01977,0) (1.75551,0)\n"
                 "(2.10247,0) (2.55646,0) (1.97654,0) (1.39927,0)\n"
                 "(0.800451,0) (0.648569,0) (1.69099,0) (1.76597,0)\n"
                 "(0.498881,0) (1.05119,0) (0.753502,0) (2.03447,0)\n");
  }
}

TEST(Tensor, eigenvalues) {
  const auto vecToStr = [](const auto &vec) {
    auto it = vec.cbegin();
    if (it == vec.cend())
      return std::string("[]");

    std::stringstream ss;
    ss << "[(" << it->real() << ", " << it->imag() << ")";
    while (++it != vec.cend())
      ss << ", (" << it->real() << ", " << it->imag() << ")";
    ss << "]";
    return ss.str();
  };
  {
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    auto data = m2.get_data(cudaq::complex_matrix::order::row_major);

    std::vector<cudaq::complex_matrix::value_type> expected_eigenvals = {
        std::complex<double>(1, 0), std::complex<double>(5, 0)};
    EXPECT_EQ(vecToStr(expected_eigenvals), vecToStr(m2.eigenvalues()));
    EXPECT_EQ(data, m2.get_data(cudaq::complex_matrix::order::row_major));
    data = m2.get_data(cudaq::complex_matrix::order::column_major);
    EXPECT_EQ(vecToStr(expected_eigenvals), vecToStr(m2.eigenvalues()));
    EXPECT_EQ(data, m2.get_data(cudaq::complex_matrix::order::column_major));
  }
}

TEST(Tensor, get_data) {
  {
    cudaq::complex_matrix::value_type *data;
    cudaq::complex_matrix m2({2., 1., 3., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m3({3., 2., 1., 4.}, {2, 2},
                             cudaq::complex_matrix::order::row_major);

    string_equal(m2.to_string(), "(2,0) (1,0)\n(3,0) (4,0)\n");
    string_equal(m3.to_string(), "(3,0) (2,0)\n(1,0) (4,0)\n");
    string_equal((m2 * m3).to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");

    data = m2.get_data(cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m4(
        std::vector<cudaq::complex_matrix::value_type>(data, data + m2.size()),
        {2, 2}, cudaq::complex_matrix::order::row_major);
    string_equal(m2.to_string(), "(2,0) (1,0)\n(3,0) (4,0)\n");
    string_equal(m4.to_string(), "(2,0) (3,0)\n(1,0) (4,0)\n");
    string_equal((m2 * m3).to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");

    data = m3.get_data(cudaq::complex_matrix::order::column_major);
    cudaq::complex_matrix m5(
        std::vector<cudaq::complex_matrix::value_type>(data, data + m2.size()),
        {2, 2}, cudaq::complex_matrix::order::row_major);
    string_equal(m3.to_string(), "(3,0) (2,0)\n(1,0) (4,0)\n");
    string_equal(m5.to_string(), "(3,0) (1,0)\n(2,0) (4,0)\n");
    string_equal((m2 * m3).to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");

    data = m2.get_data(cudaq::complex_matrix::order::row_major);
    cudaq::complex_matrix m6(
        std::vector<cudaq::complex_matrix::value_type>(data, data + m2.size()),
        {2, 2}, cudaq::complex_matrix::order::row_major);
    string_equal(m2.to_string(), "(2,0) (1,0)\n(3,0) (4,0)\n");
    string_equal(m6.to_string(), "(2,0) (1,0)\n(3,0) (4,0)\n");
    string_equal((m2 * m3).to_string(), "(7,0) (8,0)\n(13,0) (22,0)\n");

    EXPECT_EQ(data, m2.get_data(cudaq::complex_matrix::order::row_major));
    EXPECT_NE(data, m2.get_data(cudaq::complex_matrix::order::column_major));
  }
}
