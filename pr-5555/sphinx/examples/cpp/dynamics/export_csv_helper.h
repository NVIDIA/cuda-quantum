/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
inline void
export_csv_helper(std::vector<std::string> & /*headers*/,
                  std::vector<const std::vector<double> *> & /*columns*/) {}

template <typename... Rest>
inline void export_csv_helper(std::vector<std::string> &headers,
                              std::vector<const std::vector<double> *> &columns,
                              const std::string &header,
                              const std::vector<double> &column,
                              const Rest &...rest) {
  headers.push_back(header);
  columns.push_back(&column);
  export_csv_helper(headers, columns, rest...);
}

template <typename... Args>
void export_csv(const std::string &filename, const std::string &header1,
                const std::vector<double> &col1, const Args &...args) {
  static_assert(sizeof...(args) % 2 == 0,
                "Parameters must be provided in header/vector pairs.");

  std::vector<std::string> headers;
  std::vector<const std::vector<double> *> columns;
  headers.push_back(header1);
  columns.push_back(&col1);
  export_csv_helper(headers, columns, args...);

  size_t n = columns.front()->size();
  for (const auto *column : columns) {
    if (column->size() != n) {
      std::cerr << "Error: all columns must have the same length." << std::endl;
      return;
    }
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < headers.size(); ++i) {
    file << headers[i];
    if (i < headers.size() - 1) {
      file << ",";
    }
  }
  file << std::endl;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < columns.size(); ++j) {
      file << std::fixed << std::setprecision(8) << (*columns[j])[i];
      if (j < columns.size() - 1) {
        file << ",";
      }
    }
    file << std::endl;
  }
  file.close();
}
