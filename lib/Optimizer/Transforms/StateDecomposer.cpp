/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "StateDecomposer.h"

namespace cudaq::details {

std::vector<std::size_t> grayCode(std::size_t numBits) {
  std::vector<std::size_t> result(1ULL << numBits);
  for (std::size_t i = 0; i < (1ULL << numBits); ++i)
    result[i] = ((i >> 1) ^ i);
  return result;
}

std::vector<std::size_t> getControlIndices(std::size_t numBits) {
  auto code = grayCode(numBits);
  std::vector<std::size_t> indices;
  for (auto i = 0u; i < code.size(); ++i) {
    // The position of the control in the lth CNOT gate is set to match
    // the position where the lth and (l + 1)th bit strings g[l] and g[l+1] of
    // the binary reflected Gray code differ.
    auto position = std::log2(code[i] ^ code[(i + 1) % code.size()]);
    // N.B: In CUDA Quantum we write the least significant bit (LSb) on the left
    //
    //  lsb -v
    //       001
    //         ^- msb
    //
    // Meaning that the bitstring 001 represents the number four instead of one.
    // The above position calculation uses the 'normal' convention of writing
    // numbers with the LSb on the left.
    //
    // Now, what we need to find out is the position of the 1 in the bitstring.
    // If we take LSb as being position 0, then for the normal convention its
    // position will be 0. Using CUDA Quantum convention it will be 2. Hence,
    // we need to convert the position we find using:
    //
    // numBits - position - 1
    //
    // The extra -1 is to account for indices starting at 0. Using the above
    // examples:
    //
    // bitstring: 001
    // numBits: 3
    // position: 0
    //
    // We have the converted position: 2, which is what we need.
    indices.emplace_back(numBits - position - 1);
  }
  return indices;
}

std::vector<double> convertAngles(const std::span<double> alphas) {
  // Implements Eq. (3) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  //
  // N.B: The paper does fails to explicitly define what is the dot operator in
  // the exponent of -1. Ref. 3 solves the mystery: its the bitwise inner
  // product.
  auto bitwiseInnerProduct = [](std::size_t a, std::size_t b) {
    auto product = a & b;
    auto sumOfProducts = 0;
    while (product) {
      sumOfProducts += product & 0b1 ? 1 : 0;
      product = product >> 1;
    }
    return sumOfProducts;
  };
  std::vector<double> thetas(alphas.size(), 0);
  for (std::size_t i = 0u; i < alphas.size(); ++i) {
    for (std::size_t j = 0u; j < alphas.size(); ++j)
      thetas[i] +=
          bitwiseInnerProduct(j, ((i >> 1) ^ i)) & 0b1 ? -alphas[j] : alphas[j];
    thetas[i] /= alphas.size();
  }
  return thetas;
}

std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {
  // Implements Eq. (5) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  std::vector<double> angles;
  double divisor = static_cast<double>(1ULL << (k - 1));
  for (std::size_t j = 1; j <= (1ULL << (numQubits - k)); ++j) {
    double angle = 0.0;
    for (std::size_t l = 1; l <= (1ULL << (k - 1)); ++l)
      // N.B: There is an extra '-1' on these indices computations to account
      // for the fact that our indices start at 0.
      angle += data[(2 * j - 1) * (1 << (k - 1)) + l - 1] -
               data[(2 * j - 2) * (1 << (k - 1)) + l - 1];
    angles.push_back(angle / divisor);
  }
  return angles;
}

std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {
  // Implements Eq. (8) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  // N.B: There is an extra '-1' on these indices computations to account for
  // the fact that our indices start at 0.
  std::vector<double> angles;
  for (std::size_t j = 1; j <= (1ULL << (numQubits - k)); ++j) {
    double numerator = 0;
    for (std::size_t l = 1; l <= (1ULL << (k - 1)); ++l) {
      numerator +=
          std::pow(std::abs(data[(2 * j - 1) * (1 << (k - 1)) + l - 1]), 2);
    }

    double denominator = 0;
    for (std::size_t l = 1; l <= (1ULL << k); ++l) {
      denominator += std::pow(std::abs(data[(j - 1) * (1 << k) + l - 1]), 2);
    }

    if (denominator == 0.0) {
      assert(numerator == 0.0 &&
             "If the denominator is zero, the numerator must also be zero.");
      angles.push_back(0.0);
      continue;
    }
    angles.push_back(2.0 * std::asin(std::sqrt(numerator / denominator)));
  }
  return angles;
}
} // namespace cudaq::details