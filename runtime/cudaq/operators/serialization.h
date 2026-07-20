/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <cstdint>
#include <fstream>
#include <limits>

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
    const std::streamoff file_size = input.tellg();
    if (file_size <= 0)
      throw std::runtime_error("Serialized spin operator file is empty or has "
                               "an invalid size.");

    const auto byte_count = static_cast<std::uintmax_t>(file_size);
    if (byte_count % sizeof(double) != 0 ||
        byte_count > std::numeric_limits<std::size_t>::max() ||
        byte_count > static_cast<std::uintmax_t>(
                         std::numeric_limits<std::streamsize>::max()))
      throw std::runtime_error(
          "Serialized spin operator file size is invalid.");

    input.seekg(0, std::ios_base::beg);
    if (input.fail())
      throw std::runtime_error("Failed to seek serialized spin operator file.");

    const auto size = static_cast<std::size_t>(byte_count);
    std::vector<double> input_vec(size / sizeof(double));
    input.read(reinterpret_cast<char *>(input_vec.data()),
               static_cast<std::streamsize>(size));
    if (input.fail() || input.gcount() != static_cast<std::streamsize>(size))
      throw std::runtime_error("Failed to read serialized spin operator file.");

    return spin_op(input_vec);
  }
};
} // namespace cudaq
