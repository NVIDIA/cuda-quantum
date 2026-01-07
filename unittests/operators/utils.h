/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"

namespace utils {

void print(cudaq::complex_matrix mat, std::string name = "");

void assert_product_equal(
    const cudaq::product_op<cudaq::matrix_handler> &got,
    const std::complex<double> &expected_coefficient,
    const std::vector<cudaq::matrix_handler> &expected_terms);

void checkEqual(cudaq::complex_matrix a, cudaq::complex_matrix b);
void checkEqual(const cudaq::complex_matrix &denseMat,
                const cudaq::mdiag_sparse_matrix &diaMat);
cudaq::complex_matrix zero_matrix(std::size_t size);

cudaq::complex_matrix id_matrix(std::size_t size);

cudaq::complex_matrix annihilate_matrix(std::size_t size);

cudaq::complex_matrix create_matrix(std::size_t size);

cudaq::complex_matrix position_matrix(std::size_t size);

cudaq::complex_matrix momentum_matrix(std::size_t size);

cudaq::complex_matrix number_matrix(std::size_t size);

cudaq::complex_matrix parity_matrix(std::size_t size);

cudaq::complex_matrix displace_matrix(std::size_t size,
                                      std::complex<double> amplitude);

cudaq::complex_matrix squeeze_matrix(std::size_t size,
                                     std::complex<double> amplitude);

cudaq::complex_matrix PauliX_matrix();

cudaq::complex_matrix PauliZ_matrix();

cudaq::complex_matrix PauliY_matrix();

} // namespace utils
