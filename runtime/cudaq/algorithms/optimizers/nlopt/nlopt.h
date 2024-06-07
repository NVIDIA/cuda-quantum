/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/optimizer.h"

namespace cudaq::optimizers {
struct max_eval;
struct initial_parameters;

class base_nlopt : public cudaq::optimizer {
public:
  std::optional<int> max_eval;
  std::optional<std::vector<double>> initial_parameters;
  std::optional<std::vector<double>> lower_bounds;
  std::optional<std::vector<double>> upper_bounds;
  std::optional<double> f_tol;

  std::string __getstate__() const { return serialize(); }

  void __setstate__(const std::string &data) { deserialize(data); }

  std::string serialize() const {
    std::ostringstream oss;
    serializeOptional(oss, max_eval);
    serializeOptional(oss, initial_parameters);
    serializeOptional(oss, lower_bounds);
    serializeOptional(oss, upper_bounds);
    serializeOptional(oss, f_tol);
    return oss.str();
  }

  void deserialize(const std::string &data) {
    std::istringstream iss(data);
    deserializeOptional(iss, max_eval);
    deserializeOptional(iss, initial_parameters);
    deserializeOptional(iss, lower_bounds);
    deserializeOptional(iss, upper_bounds);
    deserializeOptional(iss, f_tol);
  }

protected:
  template <typename T>
  void serializeOptional(std::ostringstream &oss,
                         const std::optional<T> &opt) const {
    if (opt.has_value()) {
      oss << "1 ";
      serialize(oss, opt.value());
    } else {
      oss << "0 ";
    }
  }

  template <typename T>
  void deserializeOptional(std::istringstream &iss, std::optional<T> &opt) {
    int has_value;
    iss >> has_value;
    if (has_value) {
      T value;
      deserialize(iss, value);
      if constexpr (std::is_fundamental_v<T>) {
        opt = value;
      } else {
        opt.emplace(std::move(value));
      }
    } else {
      opt.reset();
    }
  }

  void serialize(std::ostringstream &oss,
                 const std::vector<double> &vec) const {
    oss << vec.size() << " ";
    for (const auto &val : vec) {
      oss << val << " ";
    }
  }

  void deserialize(std::istringstream &iss, std::vector<double> &vec) {
    size_t size;
    iss >> size;
    vec.resize(size);
    for (auto &val : vec) {
      iss >> val;
    }
  }

  void serialize(std::ostringstream &oss, const int &value) const {
    oss << value << " ";
  }

  void deserialize(std::istringstream &iss, int &value) { iss >> value; }

  void serialize(std::ostringstream &oss, const double &value) const {
    oss << value << " ";
  }

  void deserialize(std::istringstream &iss, double &value) { iss >> value; }
};

#define CUDAQ_NLOPT_ALGORITHM_TYPE(NAME, NEEDS_GRADIENTS)                      \
  class NAME : public base_nlopt {                                             \
  public:                                                                      \
    NAME();                                                                    \
    virtual ~NAME();                                                           \
    bool requiresGradients() override { return NEEDS_GRADIENTS; }              \
    optimization_result                                                        \
    optimize(const int dim, optimizable_function &&opt_function) override;     \
  };

CUDAQ_NLOPT_ALGORITHM_TYPE(cobyla, false)
CUDAQ_NLOPT_ALGORITHM_TYPE(neldermead, false)

} // namespace cudaq::optimizers
