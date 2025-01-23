/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace cudaq {

struct RecordLogger {
private:
  bool emitLabel = false;
  std::stringstream ss;

public:
  void emitHeader() {
    ss << "HEADER\tschema_name\t" << (emitLabel ? "labeled" : "ordered")
       << "\n";
    ss << "HEADER\tschema_version\t1.0\n";
  }

  template <typename T>
  void logSingleRecord(const T &record, std::string label = "") {
    ss << "OUTPUT\t";
    if (typeid(T) == typeid(bool))
      ss << "BOOL\t" << (record ? "true" : "false");
    else if (typeid(T) == typeid(int))
      ss << "INT\t" << record;
    else if (typeid(T) == typeid(float))
      ss << "FLOAT\t" << record;
    else if (typeid(T) == typeid(double))
      ss << "DOUBLE\t" << record;
    else
      throw std::runtime_error("Unsupported output record type");
    if (emitLabel) {
      if (label.empty())
        throw std::runtime_error(
            "Non-empty label expected for the output record");
      else
        ss << "\t" << label;
    }
    ss << "\n";
  }

  template <typename T>
  void log(const std::vector<T> &records) {
    emitHeader();
    for (auto r : records) {
      ss << "START\n";
      logSingleRecord<T>(r);
      ss << "END\t0\n"; // Assumes success always
    }
  }

  std::string getLog() { return ss.str(); }

  void writeToFile(std::ofstream &outFile) { outFile << ss.rdbuf(); }
};

} // namespace cudaq
