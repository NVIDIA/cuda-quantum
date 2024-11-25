/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <functional>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {
class operator_helpers {
public:
  // Aggregate parameters from a list of mappings
  static std::map<std::string, std::string> aggregateParameters(
      const std::vector<std::map<std::string, std::string>> &parameterMappings);

  // Extract Documentation for a specific parameter
  static std::string parameterDocs(const std::string &paramName,
                                   const std::string &docs);

  // Extract arguments from keyword arguments
  static std::pair<std::vector<std::string>, std::map<std::string, std::string>>
  argsFromKwargs(std::function<void()> fct,
                 std::map<std::string, std::string> &kwargs);

  // Generate all possible states for the given degrees and dimensions
  static std::vector<std::string>
  generateAllStates(const std::vector<int> &degrees,
                    const std::map<int, int> &dimensions);

  // Permutes a matrix according to the given permutation
  static void permuteMatrix(Eigen::MatrixXcd &matrix,
                            const std::vector<int> &permutation);

  // Converts a cudaq::ComplexMatrix to Eigen::MatrixXcd
  static Eigen::MatrixXcd cmatrixToNparray(const cudaq::complex_matrix &matrix);

  // Returns the degrees in canonical order (sorted in reverse)
  static std::vector<int> canonicalizeDegrees(const std::vector<int> &degress);
};
} // namespace cudaq