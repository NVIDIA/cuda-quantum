/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"

inline std::complex<double> &access(std::complex<double> *p,
                                    cudaq::matrix_2::Dimensions sizes,
                                    unsigned row, unsigned col) {
  return p[row * sizes.second + col];
}

cudaq::matrix_2 &cudaq::matrix_2::operator*=(const cudaq::matrix_2 &right) {
  auto new_data =
      new std::complex<double>[dimensions.first * right.dimensions.second];
  for (unsigned i = 0; i < dimensions.first; i++)
    for (unsigned j = 0; j < right.dimensions.second; j++)
      for (unsigned k = 0; k < dimensions.second; k++)
        access(new_data, right.dimensions, i, j) +=
            access(data, dimensions, i, k) *
            access(right.data, right.dimensions, k, j);
  swap(new_data);
  return *this;
}

cudaq::matrix_2 cudaq::operator*(std::complex<double> scalar,
                                 const cudaq::matrix_2 &right) {
  auto new_data = new std::complex<double>[right.dimensions.first *
                                           right.dimensions.second];
  for (unsigned i = 0; i < right.dimensions.first; i++)
    for (unsigned j = 0; j < right.dimensions.second; j++)
      access(new_data, right.dimensions, i, j) =
          scalar * access(right.data, right.dimensions, i, j);
  return {new_data, right.dimensions};
}

cudaq::matrix_2 &cudaq::matrix_2::operator+=(const cudaq::matrix_2 &right) {
  // TODO: implement me
  return *this;
}

cudaq::matrix_2 &cudaq::matrix_2::operator-=(const cudaq::matrix_2 &right) {
  // TODO: implement me
  return *this;
}

cudaq::matrix_2 &
cudaq::matrix_2::kronecker_inplace(const cudaq::matrix_2 &right) {
  // TODO: implement me
  return *this;
}

void cudaq::matrix_2::dump() {
  // TODO: implement me
}
