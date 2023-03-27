/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * This program and the accompanying materials are made available under the
 * terms of the MIT License which accompanies this distribution.
 ******************************************************************************/
#pragma once

#include <complex>
#include <memory>
#include <vector>

namespace cudaq {

class complex_matrix {
public:
  using value_type = std::complex<double>;

protected:
  std::unique_ptr<value_type> internalData;
  std::size_t nRows = 0;
  std::size_t nCols = 0;

public:
 complex_matrix(const std::size_t rows,
                 const std::size_t cols);
  complex_matrix(value_type *rawData, const std::size_t rows,
                 const std::size_t cols);
  value_type *data() { return internalData.get(); }

  complex_matrix operator*(complex_matrix& other);

  value_type& operator()(std::size_t i, std::size_t j);
  value_type minimal_eigenvalue();
  std::vector<value_type> eigenvalues();
  void set_zero();
  void dump();
};
} // namespace cudaq