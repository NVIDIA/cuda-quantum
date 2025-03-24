/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <chrono>
#include <cudaq.h>
#include <cudaq/spin_op.h>
#include <iostream>
#include <vector>

int main() {

  bool run_old = false;
  bool run_new = true;

  std::vector<int> repetitions = {100, 1000, 10000};

  // multiplication inplace with itself
  std::cout << std::endl << "multiplication inplace with itself" << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::spin_op_term prod_op = cudaq::spin_op::identity();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op *= cudaq::spin_op::x(0);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        prod_op *= cudaq::spin_op::x(0);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  std::cout << std::endl << "multiplication inplace with other" << std::endl;
  // multiplication inplace with other

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::spin_op_term prod_op = cudaq::spin_op::identity();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op *= cudaq::spin_op::x(i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        prod_op *= cudaq::spin_op::x(i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // multiplication inplace with other (reverse order)
  std::cout << std::endl
            << "multiplication inplace with other (reverse order)" << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::spin_op_term prod_op = cudaq::spin_op::identity();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op *= cudaq::spin_op::x(nr_reps - i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        prod_op *= cudaq::spin_op::x(nr_reps - i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // addition inplace with itself
  std::cout << std::endl << "addition inplace with itself" << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::sum_op op_sum = cudaq::spin_op::empty();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op += cudaq::spin_op::x(0);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum += cudaq::spin_op::x(0);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // addition inplace with other
  std::cout << std::endl << "addition inplace with other" << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::sum_op op_sum = cudaq::spin_op::empty();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      if (nr_reps > 1000)
        std::cout << "Old setup takes minutes - skipped" << std::endl;
      else {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < nr_reps; ++i)
          spin_op += cudaq::spin_op::x(i);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        std::cout << "Old setup took " << duration.count() << " seconds.\n";
      }
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum += cudaq::spin_op::x(i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // addition inplace with other (reverse order)
  std::cout << std::endl
            << "addition inplace with other (reverse order)" << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::sum_op op_sum = cudaq::spin_op::empty();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      if (nr_reps > 1000)
        std::cout << "Old setup takes minutes - skipped" << std::endl;
      else {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < nr_reps; ++i)
          spin_op += cudaq::spin_op::x(nr_reps - i);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        std::cout << "Old setup took " << duration.count() << " seconds.\n";
      }
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum += cudaq::spin_op::x(nr_reps - i);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // addition inplace with product self
  std::cout << std::endl << "addition inplace with product self" << std::endl;

  for (auto nr_reps : repetitions) {

    auto spin_prod = cudaq::spin_op();
    for (auto i = 0; i < 100; ++i)
      spin_prod *= cudaq::spin_op::x(i);
    cudaq::spin_op spin_op = spin_prod;
    auto prod_op = cudaq::spin_op::identity();
    for (auto i = 0; i < 100; ++i)
      prod_op *= cudaq::spin_op::x(i);
    cudaq::sum_op op_sum = prod_op;

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op += spin_prod;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum += prod_op;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // addition inplace with product self (reverse order)
  std::cout << std::endl
            << "addition inplace with product self (reverse order)"
            << std::endl;

  for (auto nr_reps : repetitions) {

    auto spin_prod = cudaq::spin_op();
    for (auto i = 0; i < 100; ++i)
      spin_prod *= cudaq::spin_op::x(100 - i);
    cudaq::spin_op spin_op = spin_prod;
    auto prod_op = cudaq::spin_op::identity();
    for (auto i = 0; i < 100; ++i)
      prod_op *= cudaq::spin_op::x(100 - i);
    cudaq::sum_op op_sum = prod_op;

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op += spin_prod;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum += prod_op;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // product inplace with 2-term sum on fixed degrees
  std::cout << std::endl
            << "product inplace with 2-term sum on fixed degrees" << std::endl;

  for (auto nr_reps : repetitions) {

    auto spin_term = cudaq::spin_op::x(0) + cudaq::spin_op::y(1);
    cudaq::spin_op spin_op = spin_term;
    auto prod_term = cudaq::spin_op::x(0) + cudaq::spin_op::y(1);
    cudaq::sum_op op_sum = prod_term;

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op *= spin_term;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum *= prod_term;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  repetitions = {10, 20}; // we run out of memory if we go to 30 here

  // product inplace with 2-term sum on varying degrees
  std::cout << std::endl
            << "product inplace with 2-term sum on varying degrees"
            << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::sum_op op_sum = cudaq::spin_op::empty();
    op_sum = cudaq::spin_op::identity();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        spin_op *= (cudaq::spin_op::x(i) + cudaq::spin_op::z(i + 1));
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "Old setup took " << duration.count() << " seconds.\n";
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum *= (cudaq::spin_op::x(i) + cudaq::spin_op::z(i + 1));
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // product inplace with 2-term sum on varying degrees (reverse order)
  std::cout
      << std::endl
      << "product inplace with 2-term sum on varying degrees (reverse order)"
      << std::endl;

  for (auto nr_reps : repetitions) {

    cudaq::spin_op spin_op = cudaq::spin_op();
    cudaq::sum_op op_sum =
        cudaq::spin_op::empty(); // fixme: only const & constructor is public
    op_sum = cudaq::spin_op::identity();

    std::cout << "nr ops:  " << nr_reps << std::endl;

    if (run_old) {
      /*
      start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
          spin_op *= (cudaq::spin_op::x(nr_reps - i) + cudaq::spin_op::z(nr_reps
      - i - 1)); stop = std::chrono::high_resolution_clock::now(); duration =
      std::chrono::duration<double>(stop - start); std::cout << "Old setup took
      " << duration.count() << " seconds.\n";
      */
      std::cout << "Old setup segfaults" << std::endl;
    }

    if (run_new) {
      auto start = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < nr_reps; ++i)
        op_sum *= (cudaq::spin_op::x(nr_reps - i) +
                   cudaq::spin_op::z(nr_reps - i - 1));
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "New setup took " << duration.count() << " seconds.\n";
    }
  }

  // sum of products of random terms
  std::cout << std::endl << "sum of products of random terms" << std::endl;

  std::vector<int> nr_sum_terms = {10, 100, 1000};
  std::vector<int> nr_product_terms = {100, 1000, 10000};
  auto nr_degrees = 100;
  std::vector<cudaq::spin_op> spin_ops;
  for (auto i = 0; i < nr_degrees; ++i) {
    spin_ops.push_back(cudaq::spin_op::x(i));
    spin_ops.push_back(cudaq::spin_op::y(i));
    spin_ops.push_back(cudaq::spin_op::z(i));
    spin_ops.push_back(cudaq::spin_op::i(i));
  }
  std::vector<cudaq::spin_op_term> leaf_ops;
  for (auto i = 0; i < nr_degrees; ++i) {
    leaf_ops.push_back(cudaq::spin_op::x(i));
    leaf_ops.push_back(cudaq::spin_op::y(i));
    leaf_ops.push_back(cudaq::spin_op::z(i));
    leaf_ops.push_back(cudaq::spin_op::i(i));
  }
  srand(5); // random number seed

  for (auto nr_sums : nr_sum_terms) {
    for (auto nr_prods : nr_product_terms) {

      std::vector<std::vector<int>> indices;
      for (auto i = 0; i < nr_sums; ++i) {
        indices.push_back({});
        for (auto j = 0; j < nr_prods; ++j)
          indices[i].push_back(rand() % 400);
      }

      std::cout << "nr terms " << nr_sums << ", term length " << nr_prods
                << std::endl;

      auto spin_op = cudaq::spin_op();
      auto op_sum = cudaq::spin_op::empty();

      if (run_old) {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < nr_sums; ++i) {
          auto term = cudaq::spin_op();
          for (auto j = 0; j < nr_prods; ++j)
            term *= spin_ops[indices[i][j]];
          spin_op += term;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        std::cout << "Old setup took " << duration.count() << " seconds.\n";
      }

      if (run_new) {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < nr_sums; ++i) {
          auto term = cudaq::spin_op::identity();
          for (auto j = 0; j < nr_prods; ++j)
            term *= leaf_ops[indices[i][j]];
          op_sum += term;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        std::cout << "New setup took " << duration.count() << " seconds.\n";
      }
    }
  }
  return 0;
}