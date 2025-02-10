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

int main() {

    // multiplication inplace with itself

    cudaq::spin_op spin_op = cudaq::spin::i(0);
    cudaq::product_operator<cudaq::spin_operator> prod_op = cudaq::spin_operator::i(0);

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

    spin_op = cudaq::spin::i(0);
    prod_op = cudaq::spin_operator::i(0);

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

    // addition inplace with itself

    spin_op = cudaq::spin::i(0);
    // auto op_sum = cudaq::operator_sum(cudaq::spin_operator::i(0));   // fixme: protected
    // cudaq::operator_sum op_sum = cudaq::spin_operator::i(0);         // fixme: protected
    prod_op = cudaq::spin_operator::i(0);
    cudaq::operator_sum op_sum = prod_op;

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

    spin_op = cudaq::spin::i(0);
    prod_op = cudaq::spin_operator::i(0);
    op_sum = prod_op;

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


    // addition inplace with product self

    auto spin_prod = cudaq::spin::i(0);
    for (auto i = 0; i < 100; ++i)
        spin_prod *= cudaq::spin::x(i);
    spin_op = spin_prod;
    prod_op = cudaq::spin_operator::i(0);
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

    spin_op = cudaq::spin::i(0);
    prod_op = cudaq::spin_operator::i(0);
    op_sum = prod_op;

    nr_reps = 20;
    std::cout << "product inplace with 2-term sum on varying degrees" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < nr_reps; ++i)
        spin_op *= cudaq::spin::x(i) + cudaq::spin::z(i + 1);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(stop - start);
    std::cout << "Old setup took " << duration.count() << " seconds.\n";

    start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < nr_reps; ++i)
        op_sum *= cudaq::spin_operator::x(i) + cudaq::spin_operator::z(i + 1);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(stop - start);
    std::cout << "New setup took " << duration.count() << " seconds.\n";

    return 0;
}