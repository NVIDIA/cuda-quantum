/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/gradient.h"

namespace cudaq::gradients {

/// @brief Compute the first order forward difference approximation for the
/// gradient
class forward_difference : public gradient {
public:
  using gradient::gradient;
  double step = 1e-4;

  /// @brief Compute the `forward_difference` gradient
  void compute(const std::vector<double> &x, std::vector<double> &dx,
               const spin_op &h, double funcAtX) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      auto px = getExpectedValue(tmpX, h);
      // return value back to x_i
      tmpX[i] -= step;
      dx[i] = (px - funcAtX) / step;
    }
  }

  /// @brief Compute the `forward_difference` gradient for the arbitary
  /// function, `func`, passed in by the user.
  std::vector<double>
  compute(const std::vector<double> &x,
          const std::function<double(std::vector<double>)> &func,
          double funcAtX) override {
    std::vector<double> dx(x.size());
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      double px = func(tmpX);
      // return value back to x_i
      tmpX[i] -= step;
      dx[i] = (px - funcAtX) / step;
    }
    return dx;
  }
};
} // namespace cudaq::gradients
