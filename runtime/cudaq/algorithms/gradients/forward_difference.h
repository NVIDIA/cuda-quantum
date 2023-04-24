/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/algorithms/gradient.h"

namespace cudaq::gradients {

class forward_difference : public gradient {
public:
  using gradient::gradient;
  double step = 1e-4;

  void compute(const std::vector<double> &x, std::vector<double> &dx,
               spin_op &h, double exp_h) override {
    auto tmpX = x;
    auto fx = getExpectedValue(tmpX, h);
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      auto px = getExpectedValue(tmpX, h);
      // return value back to x_i
      tmpX[i] -= step;
      dx[i] = (px - fx) / step;
    }
  }

  /// @brief Compute the `forward_difference` gradient for the arbitary
  /// function, `func`, passed in by the user.
  std::vector<double>
  compute(const std::vector<double> &x,
          std::function<double(std::vector<double>)> &func) override {
    std::vector<double> dx(x.size());
    auto tmpX = x;
    auto fx = func(tmpX);
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      double px = func(tmpX);
      // return value back to x_i
      tmpX[i] -= step;
      dx[i] = (px - fx) / step;
    }
    return dx;
  }
};
} // namespace cudaq::gradients

