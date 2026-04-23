/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <fstream>

namespace cudaq {
class spin_op_reader {
public:
  virtual ~spin_op_reader() = default;
  virtual spin_op read(const std::string &data_filename) = 0;
};

class binary_spin_op_reader : public spin_op_reader {
public:
  spin_op read(const std::string &data_filename) override {
    std::ifstream input(data_filename, std::ios::binary);
    if (input.fail())
      throw std::runtime_error(data_filename + " does not exist.");

    input.seekg(0, std::ios_base::end);
    std::size_t size = input.tellg();
    input.seekg(0, std::ios_base::beg);
    std::vector<double> input_vec(size / sizeof(double));
    input.read((char *)&input_vec[0], size);
    return spin_op(input_vec);
  }

  [[deprecated("overload provided for compatibility with the deprecated "
               "serialization format - please migrate to the new format and "
               "use the overload read(const std::string &)")]] spin_op
  read(const std::string &data_filename, bool legacy_format) {
    if (!legacy_format)
      return read(data_filename);

    std::ifstream input(data_filename, std::ios::binary);
    if (input.fail())
      throw std::runtime_error(data_filename + " does not exist.");

    input.seekg(0, std::ios_base::end);
    std::size_t size = input.tellg();
    input.seekg(0, std::ios_base::beg);
    std::vector<double> input_vec(size / sizeof(double));
    input.read((char *)&input_vec[0], size);
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    auto n_terms = (std::size_t)input_vec.back();
    auto nQubits = (input_vec.size() - 1 - 2 * n_terms) / n_terms;
    return spin_op(input_vec, nQubits);
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
  }
};
} // namespace cudaq
