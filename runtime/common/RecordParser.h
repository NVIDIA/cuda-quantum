/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/cudaq_utils.h"
#include <string>
#include <vector>

namespace cudaq {

struct OutputRecord {
  void *buffer;
  std::size_t size;
};

struct RecordParser {
private:
  bool labelExpected = false;

public:
  std::vector<OutputRecord> parse(const std::string &data) {
    std::vector<OutputRecord> results;
    std::vector<std::string> lines = cudaq::split(data, '\n');
    for (auto line : lines) {
      std::vector<std::string> entries = cudaq::split(line, '\t');
      /// TODO: Handle labeled records
      if (!entries.empty()) {
        if (entries[0] == "OUTPUT") {
          if ("BOOL" == entries[1]) {
            bool value;
            if ("true" == entries[2])
              value = true;
            else if ("false" == entries[2])
              value = false;
            results.emplace_back(OutputRecord{
                reinterpret_cast<void *>(new bool(value)), sizeof(bool)});
          } else if ("INT" == entries[1]) {
            results.emplace_back(OutputRecord{
                reinterpret_cast<void *>(new int(std::stoi(entries[2]))),
                sizeof(int)});
          } else if ("FLOAT" == entries[1]) {
            results.emplace_back(OutputRecord{
                reinterpret_cast<void *>(new int(std::stof(entries[2]))),
                sizeof(float)});
          } else if ("DOUBLE" == entries[1]) {
            results.emplace_back(OutputRecord{
                reinterpret_cast<void *>(new int(std::stod(entries[2]))),
                sizeof(double)});
          }
          /// TODO: Handle more types
        }
      }
    }
    return results;
  }
};

} // namespace cudaq
