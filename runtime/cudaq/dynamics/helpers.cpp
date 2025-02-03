/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <ranges>
#include "cudaq/helpers.h"
#include "cudaq/cudm_error_handling.h"
#include <iostream>
#include <map>
#include <sstream>

namespace cudaq {
namespace detail {

class _OperatorHelpers {
public:
  _OperatorHelpers() = default;

  // Aggregate parameters from multiple mappings.
  std::map<std::string, std::string> aggregate_parameters(
      const std::vector<std::map<std::string, std::string>> &parameter_mappings) {
    std::map<std::string, std::string> parameter_descriptions;

    for (const auto &descriptions : parameter_mappings) {
      for (const auto &[key, new_desc] : descriptions) {
        if (!parameter_descriptions[key].empty() && !new_desc.empty()) {
          parameter_descriptions[key] += "\n---\n" + new_desc;
        } else {
          parameter_descriptions[key] = new_desc;
        }
      }
    }

    return parameter_descriptions;
  }

  // Extract documentation for a specific parameter from docstring.
  std::string parameter_docs(const std::string &param_name,
                                              const std::string &docs) {
    if (param_name.empty() || docs.empty()) {
      return "";
    }

    try {
      std::regex keyword_pattern(R"(^\s*(Arguments|Args):\s*$)",
                                std::regex::multiline);
      std::regex param_pattern(R"(^\s*)" + param_name +
                                  R"(\s*(\(.*\))?:\s*(.*)$)",
                              std::regex::multiline);

      std::smatch match;
      std::sregex_iterator it(docs.begin(), docs.end(), keyword_pattern);
      std::sregex_iterator end;

      if (it != end) {
        std::string params_section = docs.substr(it->position() + it->length());
        if (std::regex_search(params_section, match, param_pattern)) {
          std::string param_docs = match.str(2);
          return std::regex_replace(param_docs, std::regex(R"(\s+)"), " ");
        }
      }
    } catch (...) {
      return "";
    }

    return "";
  }

  // Extract positional arguments and keyword-only arguments.
  std::pair<std::vector<std::string>, std::map<std::string, std::string>>
  args_from_kwargs(
      const std::map<std::string, std::string> &kwargs,
      const std::vector<std::string> &required_args,
      const std::vector<std::string> &kwonly_args) {
    std::vector<std::string> extracted_args;
    std::map<std::string, std::string> kwonly_dict;

    for (const auto &arg : required_args) {
      if (kwargs.count(arg)) {
        extracted_args.push_back(kwargs.at(arg));
      } else {
        throw std::invalid_argument("Missing required argument: " + arg);
      }
    }

    for (const auto &arg : kwonly_args) {
      if (kwargs.count(arg)) {
        kwonly_dict[arg] = kwargs.at(arg);
      }
    }

    return {extracted_args, kwonly_dict};
  }

  /// Generates all possible states for the given dimensions ordered according
  /// to the sequence of degrees (ordering is relevant if dimensions differ).
  std::vector<std::string>
  generate_all_states(std::vector<int> degrees, std::map<int, int> dimensions) {
    if (degrees.size() == 0)
      return {};

    std::vector<std::string> states;
    int range = dimensions[degrees[0]];
    for (auto state = 0; state < range; state++) {
      states.push_back(std::to_string(state));
    }

    for (auto idx = 1; idx < degrees.size(); ++idx) {
      std::vector<std::string> result;
      for (auto current : states) {
        for (auto state = 0; state < dimensions[degrees[idx]]; state++) {
          result.push_back(current + std::to_string(state));
        }
      }
      states = result;
    }

    return states;
  }

  cudaq::matrix_2 permute_matrix(cudaq::matrix_2 matrix,
                                        std::vector<int> permutation) {
    auto result = cudaq::matrix_2(matrix.get_rows(), matrix.get_columns());
    std::vector<std::complex<double>> sorted_values;
    for (std::size_t permuted : permutation) {
      for (std::size_t permuted_again : permutation) {
        sorted_values.push_back(matrix[{permuted, permuted_again}]);
      }
    }
    int idx = 0;
    for (std::size_t row = 0; row < result.get_rows(); row++) {
      for (std::size_t col = 0; col < result.get_columns(); col++) {
        result[{row, col}] = sorted_values[idx];
        idx++;
      }
    }
    return result;
  }

  std::vector<int> canonicalize_degrees(std::vector<int> degrees) {
    std::sort(degrees.begin(), degrees.end(), std::greater<int>());
    return degrees;
  }

};
}
}
