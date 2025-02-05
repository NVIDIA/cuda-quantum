/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include "cudaq/operators.h"
#include "cudaq/dynamics/matrix_operators.h"

namespace utils {

void print(cudaq::matrix_2 mat);

void assert_product_equal(const cudaq::product_operator<cudaq::matrix_operator> &got, 
                          const std::complex<double> &expected_coefficient,
                          const std::vector<cudaq::matrix_operator> &expected_terms);

void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b);

cudaq::matrix_2 zero_matrix(std::size_t size);

cudaq::matrix_2 id_matrix(std::size_t size);

cudaq::matrix_2 annihilate_matrix(std::size_t size);

cudaq::matrix_2 create_matrix(std::size_t size);

cudaq::matrix_2 position_matrix(std::size_t size);

cudaq::matrix_2 momentum_matrix(std::size_t size);

cudaq::matrix_2 number_matrix(std::size_t size);

cudaq::matrix_2 parity_matrix(std::size_t size);

cudaq::matrix_2 displace_matrix(std::size_t size,
                                std::complex<double> amplitude);

cudaq::matrix_2 squeeze_matrix(std::size_t size,
                               std::complex<double> amplitude);

cudaq::matrix_2 PauliX_matrix();

cudaq::matrix_2 PauliZ_matrix();

} // namespace utils
