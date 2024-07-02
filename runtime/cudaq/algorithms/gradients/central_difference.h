/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/gradient.h"

namespace cudaq::gradients {

class central_difference : public gradient {
public:
  using gradient::gradient;
  static constexpr double default_step = 1e-4;
  double step = default_step;

  central_difference(double s = default_step) : gradient(), step(s) {}

  virtual std::unique_ptr<cudaq::gradient> clone() override {
    return std::make_unique<central_difference>(step);
  }

  void compute(const std::vector<double> &x, std::vector<double> &dx,
               const spin_op &h, double exp_h) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      // auto savepx = tmpX[i];
      auto px = getExpectedValue(tmpX, h);
      // decrease the value to x_i - dx_i
      tmpX[i] -= 2 * step;
      // auto savemx = tmpX[i];
      auto mx = getExpectedValue(tmpX, h);
      // return value back to x_i
      tmpX[i] += step;
      dx[i] = (px - mx) / (2. * step);
      // printf("compute: tmp[%lu]=%.16f dx[%lu]=%.16f step=%.16f px=%.16f
      // mx=%.16f "
      //        "savepx=%.16f savemx=%.16f\n",
      //        i, tmpX[i], i, dx[i], step, px, mx, savepx, savemx);
    }
  }

  /// @brief Compute the `central_difference` gradient for the arbitrary
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
      // decrease the value to x_i - dx_i
      tmpX[i] -= 2 * step;
      double mx = func(tmpX);
      // return value back to x_i
      tmpX[i] += step;
      dx[i] = (px - mx) / (2. * step);
    }
    return dx;
  }
};
} // namespace cudaq::gradients
