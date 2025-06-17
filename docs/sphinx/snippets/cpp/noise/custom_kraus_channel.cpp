/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ custom_kraus_channel.cpp && ./a.out`

#include <cudaq.h>

struct my_custom_kraus_channel_subtype : public ::cudaq::kraus_channel {
  static constexpr std::size_t num_parameters = 1;
  static constexpr std::size_t num_targets = 1;

  my_custom_kraus_channel_subtype(const std::vector<cudaq::real> &params) {
    std::vector<cudaq::complex> k0v{std::sqrt(1 - params[0]), 0, 0,
                                    std::sqrt(1 - params[0])},
        k1v{0, std::sqrt(params[0]), std::sqrt(params[0]), 0};
    push_back(cudaq::kraus_op(k0v));
    push_back(cudaq::kraus_op(k1v));
    validateCompleteness();
    generateUnitaryParameters();
  }
  REGISTER_KRAUS_CHANNEL("my_custom_kraus_channel_subtype");
};

// Example usage
int main() {
  cudaq::noise_model noise;
  noise.register_channel<my_custom_kraus_channel_subtype>();

  return 0;
}

