/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <chrono>
#include <cudaq.h>
#include <cudaq/operators.h>
#include <iostream>
#include <vector>

int main() {

  // multiplication inplace with itself

  cudaq::spin_op spin_op = cudaq::spin_op();
  cudaq::product_operator<cudaq::spin_operator> prod_op =
      cudaq::spin_operator::identity();

  int nr_reps = 1000;
  std::cout << "multiplication inplace with itself" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op *= cudaq::spin::x(0);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    prod_op *= cudaq::spin_operator::x(0);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // multiplication inplace with other

  spin_op = cudaq::spin_op();
  prod_op = cudaq::spin_operator::identity();

  nr_reps = 1000;
  std::cout << "multiplication inplace with other" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op *= cudaq::spin::x(i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    prod_op *= cudaq::spin_operator::x(i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // multiplication inplace with other (reverse order)

  spin_op = cudaq::spin_op();
  prod_op = cudaq::spin_operator::identity();

  nr_reps = 1000;
  std::cout << "multiplication inplace with other (reverse order)" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op *= cudaq::spin::x(nr_reps - i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    prod_op *= cudaq::spin_operator::x(nr_reps - i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // addition inplace with itself

  spin_op = cudaq::spin_op();
  cudaq::operator_sum op_sum = cudaq::spin_operator::empty();

  nr_reps = 1000;
  std::cout << "addition inplace with itself" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op += cudaq::spin::x(0);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum += cudaq::spin_operator::x(0);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // addition inplace with other

  spin_op = cudaq::spin_op();
  op_sum = cudaq::spin_operator::empty();

  nr_reps = 1000;
  std::cout << "addition inplace with other" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op += cudaq::spin::x(i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum += cudaq::spin_operator::x(i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // addition inplace with other (reverse order)

  spin_op = cudaq::spin_op();
  op_sum = cudaq::spin_operator::empty();

  nr_reps = 1000;
  std::cout << "addition inplace with other (reverse order)" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op += cudaq::spin::x(nr_reps - i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum += cudaq::spin_operator::x(nr_reps - i);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // addition inplace with product self

  auto spin_prod = cudaq::spin_op();
  for (auto i = 0; i < 100; ++i)
    spin_prod *= cudaq::spin::x(i);
  spin_op = spin_prod;
  prod_op = cudaq::spin_operator::identity();
  for (auto i = 0; i < 100; ++i)
    prod_op *= cudaq::spin_operator::x(i);
  op_sum = prod_op;

  nr_reps = 1000;
  std::cout << "addition inplace with product self" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op += spin_prod;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum += prod_op;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // addition inplace with product self (reverse order)

  spin_prod = cudaq::spin_op();
  for (auto i = 0; i < 100; ++i)
    spin_prod *= cudaq::spin::x(100 - i);
  spin_op = spin_prod;
  prod_op = cudaq::spin_operator::identity();
  for (auto i = 0; i < 100; ++i)
    prod_op *= cudaq::spin_operator::x(100 - i);
  op_sum = prod_op;

  nr_reps = 1000;
  std::cout << "addition inplace with product self" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op += spin_prod;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum += prod_op;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // product inplace with 2-term sum on fixed degrees

  auto spin_term = cudaq::spin::x(0) + cudaq::spin::y(1);
  spin_op = spin_term;
  auto prod_term = cudaq::spin_operator::x(0) + cudaq::spin_operator::y(1);
  op_sum = prod_term;

  nr_reps = 20;
  std::cout << "product inplace with 2-term sum on fixed degrees" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op *= spin_term;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum *= prod_term;
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // product inplace with 2-term sum on varying degrees

  spin_op = cudaq::spin_op();
  op_sum = cudaq::spin_operator::identity();

  nr_reps = 20;
  std::cout << "product inplace with 2-term sum on varying degrees"
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    spin_op *= (cudaq::spin::x(i) + cudaq::spin::z(i + 1));
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum *= (cudaq::spin_operator::x(i) + cudaq::spin_operator::z(i + 1));
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // product inplace with 2-term sum on varying degrees (reverse order)

  spin_op = cudaq::spin_op();
  op_sum = cudaq::spin_operator::identity();

  nr_reps = 20;
  std::cout
      << "product inplace with 2-term sum on varying degrees (reverse order)"
      << std::endl;

  /*
  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
      spin_op *= (cudaq::spin::x(nr_reps - i) + cudaq::spin::z(nr_reps - i -
  1)); stop = std::chrono::high_resolution_clock::now(); duration =
  std::chrono::duration<double>(stop - start); std::cout << "Old setup took " <<
  duration.count() << " seconds.\n";
  */
  std::cout << "Old setup segfaults" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_reps; ++i)
    op_sum *= (cudaq::spin_operator::x(nr_reps - i) +
               cudaq::spin_operator::z(nr_reps - i - 1));
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  // sum of products of random terms

  auto nr_degrees = 100;
  std::vector<cudaq::spin_op> spin_ops;
  for (auto i = 0; i < nr_degrees; ++i) {
    spin_ops.push_back(cudaq::spin::x(i));
    spin_ops.push_back(cudaq::spin::y(i));
    spin_ops.push_back(cudaq::spin::z(i));
    spin_ops.push_back(cudaq::spin::i(i));
  }
  std::vector<cudaq::product_operator<cudaq::spin_operator>> leaf_ops;
  for (auto i = 0; i < nr_degrees; ++i) {
    leaf_ops.push_back(cudaq::spin_operator::x(i));
    leaf_ops.push_back(cudaq::spin_operator::y(i));
    leaf_ops.push_back(cudaq::spin_operator::z(i));
    leaf_ops.push_back(cudaq::spin_operator::i(i));
  }

  auto term_length = 1000;
  auto nr_terms = 200;
  srand(5); // random number seed
  std::vector<std::vector<int>> indices;
  for (auto i = 0; i < nr_terms; ++i) {
    indices.push_back({});
    for (auto j = 0; j < term_length; ++j)
      indices[i].push_back(rand() % 400);
  }

  spin_op = cudaq::spin_op();
  op_sum = cudaq::spin_operator::empty();

  std::cout << "sum of products of random terms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_terms; ++i) {
    auto term = cudaq::spin_op();
    for (auto j = 0; j < term_length; ++j)
      term *= spin_ops[indices[i][j]];
    spin_op += term;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "Old setup took " << duration.count() << " seconds.\n";

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < nr_terms; ++i) {
    auto term = cudaq::spin_operator::identity();
    for (auto j = 0; j < term_length; ++j)
      term *= leaf_ops[indices[i][j]];
    op_sum += term;
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(stop - start);
  std::cout << "New setup took " << duration.count() << " seconds.\n";

  return 0;
}