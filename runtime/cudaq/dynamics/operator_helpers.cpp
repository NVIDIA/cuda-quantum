/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "operator_helpers.h"
#include <sstream>

namespace cudaq {
std::map<std::string, std::string> operator_helpers::aggregateParameters(
    const std::vector<std::map<std::string, std::string>> &parameterMappings) {
  std::map<std::string, std::string> paramDescriptions;
  for (const auto &descriptions : parameterMappings) {
    for (const auto &[key, newDesc] : descriptions) {
      std::string &existingDesc = paramDescriptions[key];
      if (!existingDesc.empty() && !newDesc.empty()) {
        existingDesc += "\n---\n" + newDesc;
      } else if (!newDesc.empty()) {
        existingDesc = newDesc;
      }
    }
  }
  return paramDescriptions;
}

std::string operator_helpers::parameterDocs(const std::string &paramName,
                                            const std::string &docs) {
  if (paramName.empty() || docs.empty())
    return "";

  try {
    std::regex keywordPattern(R"(^\\s*(Arguments|Args):\\s*\\n)",
                              std::regex::multiline);
    std::smatch match;
    if (std::regex_search(docs, match, keywordPattern)) {
      std::string paramDocs = match.suffix();
      std::regex paramPattern(
          R"(^\s*" + paramName + R"(\s*(\(.*\))?:)\s*(.*)$)",
          std::regex::multiline);
      if (std::regex_search(paramDocs, match, paramPattern)) {
        return match.str(2);
      }
    }
  } catch (const std::exception &) {
    return "";
  }
  return "";
}

std::pair<std::vector<std::string>, std::map<std::string, std::string>>
operator_helpers::argsFromKwargs(std::function<void()> fct,
                                 std::map<std::string, std::string> &kwargs) {
  // FIXME implement later
  return {};
}

std::vector<std::string>
operator_helpers::generateAllStates(const std::vector<int> &degrees,
                                    const std::map<int, int> &dimensions) {
  if (degrees.empty())
    return std::vector<std::string>();

  std::vector<std::vector<std::string>> states;
  for (int state = 0; state < dimensions.at(degrees[0]); state++) {
    states.push_back({std::to_string(state)});
  }

  for (size_t i = 1; i < degrees.size(); i++) {
    std::vector<std::vector<std::string>> newStates;
    int d = degrees[i];
    for (const auto &current : states) {
      for (int state = 0; state < dimensions.at(d); state++) {
        std::vector<std::string> newState = current;
        newState.push_back(std::to_string(state));
        newStates.push_back(newState);
      }
    }
    states = newStates;
  }

  std::vector<std::string> result;
  for (const auto &state : states) {
    std::string combined;
    for (const auto &s : state) {
      combined += s;
    }
    result.push_back(combined);
  }
  return result;
}

// void operator_helpers::permuteMatrix(Eigen::MatrixXcd &matrix, const
// std::vector<int> &permutation) {
//     Eigen::MatrixXcd permuted = matrix(permutation,
//     Eigen::AllAtOnceTraversal); matrix = permuted;
// }

Eigen::MatrixXcd
operator_helpers::cmatrixToNparray(const cudaq::complex_matrix &cmatrix) {
  Eigen::MatrixXcd matrix(cmatrix.rows(), cmatrix.cols());
  for (int row = 0; row < cmatrix.rows(); row++) {
    for (int col = 0; col < cmatrix.cols(); col++) {
      matrix(row, col) = cmatrix(row, col);
    }
  }
  return matrix;
}

std::vector<int>
operator_helpers::canonicalizeDegrees(const std::vector<int> &degrees) {
  std::vector<int> sortedDegrees = degrees;
  std::sort(sortedDegrees.rbegin(), sortedDegrees.rend());
  return sortedDegrees;
}
} // namespace cudaq