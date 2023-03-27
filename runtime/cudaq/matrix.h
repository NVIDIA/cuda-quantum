/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include <complex>
#include <memory>
#include <vector>

namespace cudaq {

/// @brief The complex_matrix provides an abstraction for
/// describing a matrix with N rows and M columns containing
/// complex elements.
class complex_matrix {
public:
  using value_type = std::complex<double>;

protected:
  /// @brief Pointer to an array of data representing the matrix
  std::unique_ptr<value_type> internalData;

  /// @brief The number of rows in this matrix
  std::size_t nRows = 0;

  /// @brief The number of columns in this matrix
  std::size_t nCols = 0;

public:
  /// @brief Create a matrix of the given sizes, all elements
  /// initialized to 0.0
  complex_matrix(const std::size_t rows, const std::size_t cols);

  /// @brief Create a matrix from an existing data pointer.
  complex_matrix(value_type *rawData, const std::size_t rows,
                 const std::size_t cols);
  
  /// @brief Return the internal data representation
  value_type *data() { return internalData.get(); }

  /// @brief Multiply this matrix with the provided other matrix. 
  /// This does not modify this matrix but instead returns a 
  /// new matrix value. 
  complex_matrix operator*(complex_matrix &other);

  /// @brief Return the element at the ith row and jth column.
  value_type &operator()(std::size_t i, std::size_t j);
  
  /// @brief Return the minimal eigenvalue for this matrix. 
  value_type minimal_eigenvalue();

  /// @brief Return this matrix's eigenvalues.
  std::vector<value_type> eigenvalues();
  
  /// @brief Set all elements in this matrix to 0.0
  void set_zero();

  /// @brief Print this matrix to the given output stream
  void dump(std::ostream& os);

  /// @brief Print this matrix to standard out
  void dump();
};
} // namespace cudaq