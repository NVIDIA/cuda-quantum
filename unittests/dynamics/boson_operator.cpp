/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <iostream>
#include "utils.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>
#include "cudaq/dynamics/boson_operators.h"

TEST(OperatorExpressions, checkBosonOpsUnary) {
  auto op = cudaq::boson_operator::number(0);
  utils::checkEqual((-op).to_matrix({{0, 3}}), -1.0 * utils::number_matrix(3));
}

TEST(OperatorExpressions, checkPreBuiltBosonOps) {

  // number operator
  {
    auto nr_op = cudaq::boson_operator::number(0);
    for (auto d = 2; d < 7; ++d) {
      auto nr_mat = utils::number_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = nr_mat;
        auto got = nr_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= nr_mat;      
          got *= nr_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }  
  }

  // creation operator
  {
    auto ad_op = cudaq::boson_operator::create(0);
    for (auto d = 2; d < 7; ++d) {
      auto ad_mat = utils::create_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = ad_mat;
        auto got = ad_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= ad_mat;      
          got *= ad_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }  
  }

  // annihilation operator
  {
    auto a_op = cudaq::boson_operator::annihilate(0);
    for (auto d = 2; d < 7; ++d) {
      auto a_mat = utils::annihilate_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = a_mat;
        auto got = a_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= a_mat;      
          got *= a_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }  
  }

  // general in-place multiplication
  {
    auto max_nr_consecutive = 6;
    auto nr_op = cudaq::boson_operator::number(0);
    auto ad_op = cudaq::boson_operator::create(0);
    auto a_op = cudaq::boson_operator::annihilate(0);
    for (auto d = 3; d < 4; ++d) {

      // we use a larger dimension to compute the correct expected matrices
      // to ensure the expected matrix is no impacted by finite-size errors
      auto nr_mat = utils::number_matrix(d + max_nr_consecutive);
      auto ad_mat = utils::create_matrix(d + max_nr_consecutive);
      auto a_mat = utils::annihilate_matrix(d + max_nr_consecutive);

      for (auto nrs = 0; nrs < max_nr_consecutive; ++nrs) {
        for (auto ads = 0; ads < max_nr_consecutive; ++ ads) {
          for (auto as = 0; as < max_nr_consecutive; ++ as) {

            // Check Ads * Ns * As

            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# As: " << as << std::endl;

            auto padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            auto expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            auto got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // Check  Ads * As * Ns

            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# As: " << as << ", ";
            std::cout << "# Ns: " << nrs << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));            

            // Check Ns * Ads * As

            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# As: " << as << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}})); 

            // check Ns * As * Ads

            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# As: " << as << ", "; 
            std::cout << "# Ads: " << ads << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}})); 

            // check As * Ns * Ads

            std::cout << "# As: " << as << ", "; 
            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# Ads: " << ads << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}})); 

            // check As * Ads * Ns

            std::cout << "# As: " << as << ", "; 
            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# Ns: " << nrs << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            expected = cudaq::matrix_2(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_operator::identity(0);
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}})); 
          }
        }
      }
    }
  }
}