/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/helpers.h"
#include <iostream>
#include <map>
#include <sstream>

namespace cudaq {
// Aggregate parameters from multiple mappings.
std::map<std::string, std::string> OperatorHelpers::aggregate_parameters(
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
std::string OperatorHelpers::parameter_docs(const std::string &param_name,
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
OperatorHelpers::args_from_kwargs(
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

// Generate all possible quantum states for given degrees and dimensions
std::vector<std::string>
OperatorHelpers::generate_all_states(const std::vector<int> &degrees,
                                     const std::map<int, int> &dimensions) {
  if (degrees.empty()) {
    return {};
  }

  std::vector<std::vector<std::string>> states;
  for (int state = 0; state < dimensions.at(degrees[0]); state++) {
    states.push_back({std::to_string(state)});
  }

  for (size_t i = 1; i < degrees.size(); i++) {
    std::vector<std::vector<std::string>> new_states;
    for (const auto &current : states) {
      for (int state = 0; state < dimensions.at(degrees[i]); state++) {
        auto new_entry = current;
        new_entry.push_back(std::to_string(state));
        new_states.push_back(new_entry);
      }
    }
    states = new_states;
  }

  std::vector<std::string> result;
  for (const auto &state : states) {
    std::ostringstream joined;
    for (const auto &s : state) {
      joined << s;
    }
    result.push_back(joined.str());
  }
  return result;
}

// Permute a given eigen matrix
void OperatorHelpers::permute_matrix(Eigen::MatrixXcd &matrix,
                                     const std::vector<int> &permutation) {
  Eigen::MatrixXcd permuted_matrix(matrix.rows(), matrix.cols());

  for (size_t i = 0; i < permutation.size(); i++) {
    for (size_t j = 0; j < permutation.size(); j++) {
      permuted_matrix(i, j) = matrix(permutation[i], permutation[j]);
    }
  }

  matrix = permuted_matrix;
}

// Canonicalize degrees by sorting in descending order
std::vector<int>
OperatorHelpers::canonicalize_degrees(const std::vector<int> &degrees) {
  std::vector<int> sorted_degrees = degrees;
  std::sort(sorted_degrees.rbegin(), sorted_degrees.rend());
  return sorted_degrees;
}

} // namespace cudaq
