/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/tensor.h"
#include <sstream>

inline std::complex<double> &access(std::complex<double> *p,
                                    cudaq::matrix_2::Dimensions sizes,
                                    std::size_t row, std::size_t col) {
  return p[row * sizes.second + col];
}

cudaq::matrix_2 &cudaq::matrix_2::operator*=(const cudaq::matrix_2 &right) {
  if (get_columns() != right.get_rows()) {
    clear();
    return *this;
  }
  auto new_data = new std::complex<double>[get_rows() * right.get_columns()];
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < right.get_columns(); j++)
      for (std::size_t k = 0; k < get_columns(); k++)
        access(new_data, right.dimensions, i, j) +=
            access(data, dimensions, i, k) *
            access(right.data, right.dimensions, k, j);
  swap(new_data);
  return *this;
}

cudaq::matrix_2 cudaq::operator*(std::complex<double> scalar,
                                 const cudaq::matrix_2 &right) {
  auto new_data =
      new std::complex<double>[right.get_rows() * right.get_columns()];
  for (std::size_t i = 0; i < right.get_rows(); i++)
    for (std::size_t j = 0; j < right.get_columns(); j++)
      access(new_data, right.dimensions, i, j) =
          scalar * access(right.data, right.dimensions, i, j);
  return {new_data, right.dimensions};
}

cudaq::matrix_2 &cudaq::matrix_2::operator+=(const cudaq::matrix_2 &right) {
  if (!(get_rows() == right.get_rows() &&
        get_columns() == right.get_columns())) {
    clear();
    return *this;
  }
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < get_columns(); j++)
      access(data, dimensions, i, j) +=
          access(right.data, right.dimensions, i, j);
  return *this;
}

cudaq::matrix_2 &cudaq::matrix_2::operator-=(const cudaq::matrix_2 &right) {
  if (!(get_rows() == right.get_rows() &&
        get_columns() == right.get_columns())) {
    clear();
    return *this;
  }
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t j = 0; j < get_columns(); j++)
      access(data, dimensions, i, j) -=
          access(right.data, right.dimensions, i, j);
  return *this;
}

cudaq::matrix_2 &
cudaq::matrix_2::kronecker_inplace(const cudaq::matrix_2 &right) {
  Dimensions new_dim{get_rows() * right.get_rows(),
                     get_columns() * right.get_columns()};
  auto new_data = new std::complex<double>[get_rows() * right.get_rows() *
                                           get_columns() * right.get_columns()];
  for (std::size_t i = 0; i < get_rows(); i++)
    for (std::size_t k = 0; k < right.get_rows(); k++)
      for (std::size_t j = 0; j < get_columns(); j++)
        for (std::size_t m = 0; m < right.get_columns(); m++)
          access(new_data, new_dim, right.get_rows() * i + k,
                 right.get_columns() * j + m) =
              access(data, dimensions, i, j) *
              access(right.data, right.dimensions, k, m);
  swap(new_data);
  dimensions = new_dim;
  return *this;
}

std::string cudaq::matrix_2::dump() const {
  std::ostringstream out;
  out << '{';
  for (std::size_t i = 0; i < get_rows(); i++) {
    out << "  {";
    for (std::size_t j = 0; j < get_columns(); j++)
      out << ' ' << access(data, dimensions, i, j) << ' ';
    out << "}\n ";
  }
  out << '}';
  return out.str();
}
