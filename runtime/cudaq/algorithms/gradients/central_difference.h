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
  double step = 1e-4;

  void compute(const std::vector<double> &x, std::vector<double> &dx,
               const spin_op &h, double exp_h) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      auto px = getExpectedValue(tmpX, h);
      // decrease the value to x_i - dx_i
      tmpX[i] -= 2 * step;
      auto mx = getExpectedValue(tmpX, h);
      // return value back to x_i
      tmpX[i] += step;
      dx[i] = (px - mx) / (2. * step);
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

  /// @brief Serialize function
  std::string serialize() const {
    std::ostringstream oss;
    oss << step;
    return oss.str();
  }

  /// @brief Deserialize function
  central_difference deserialize(const std::string &serialized_data) {
    std::istringstream iss(serialized_data);
    double step_value;
    iss >> step_value;
    central_difference cd;
    cd.step = step_value;
    return cd;
  }

  std::string __getstate__() const { return serialize(); }

  void __setstate__(const std::string &data) { deserialize(data); }
};
} // namespace cudaq::gradients
