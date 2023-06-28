/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/gradient.h"

namespace cudaq::gradients {
class parameter_shift : public gradient {
public:
  using gradient::gradient;
  double shiftScalar = 0.5;

  void compute(const std::vector<double> &x, std::vector<double> &dx,
               spin_op &h, double exp_h) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + (shiftScalar * pi)
      tmpX[i] += shiftScalar * M_PI;
      auto px = getExpectedValue(tmpX, h);
      // decrease value to x_i - (shiftScalar * pi)
      tmpX[i] -= 2 * shiftScalar * M_PI;
      auto mx = getExpectedValue(tmpX, h);
      // return value back to x_i
      tmpX[i] += shiftScalar * M_PI;
      dx[i] = (px - mx) / 2.;
    }
  }

  /// @brief Compute the `parameter_shift` gradient for the arbitrary
  /// function, `func`, passed in by the user.
  std::vector<double>
  compute(const std::vector<double> &x,
          const std::function<double(std::vector<double>)> &func,
          double funcAtX) override {
    std::vector<double> dx(x.size());
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + (shiftScalar * pi)
      tmpX[i] += shiftScalar * M_PI;
      double px = func(tmpX);
      // decrease value to x_i - (shiftScalar * pi)
      tmpX[i] -= 2 * shiftScalar * M_PI;
      double mx = func(tmpX);
      // return value back to x_i
      tmpX[i] += shiftScalar * M_PI;
      dx[i] = (px - mx) / 2.;
    }
    return dx;
  }
};
} // namespace cudaq::gradients
