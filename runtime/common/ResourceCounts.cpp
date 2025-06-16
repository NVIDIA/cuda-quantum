/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResourceCounts.h"

#include <algorithm>
#include <numeric>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

namespace cudaq {

void resource_counts::append(GateData gate, size_t count) {
  std::string gatestr("c", gate.controls);
  gatestr += gate.name;

  append(gatestr, count);
}

void resource_counts::append(const std::string gate, size_t count) {
  auto iter = gateCounts.find(gate);
  if (iter != gateCounts.end()) {
    iter->second += count;
  } else {
    gateCounts.insert({gate, count});
  }

  totalGates += count;
}

std::size_t resource_counts::count(const GateData gate) const {
  std::string gatestr("c", gate.controls);
  gatestr += gate.name;

  auto iter = gateCounts.find(gatestr);
  if (iter != gateCounts.end()) {
    return iter->second;
  } else {
    return 0;
  }
}

std::size_t resource_counts::count(const std::string gate) const {
  auto iter = gateCounts.find(gate);
  if (iter != gateCounts.end()) {
    return iter->second;
  } else {
    return 0;
  }
}

void resource_counts::dump(std::ostream &os) const {
  os << "Total # of gates: " << totalGates;
  os << ", total # of qubits: " << numQubits;
  os << "\n";
  os << "{ ";
  os << "\n  ";
  std::size_t counter = 0;
  for (auto &result : gateCounts) {
    os << result.first << " :  " << result.second;
    bool isLast = counter == gateCounts.size() - 1;
    counter++;
    os << "\n" << (!isLast ? "  " : "");
  }
  os << "}\n";
}

void resource_counts::dump() const { dump(std::cout); }

void resource_counts::clear() {
  gateCounts.clear();
  totalGates = 0;
  numQubits = 0;
}

CountsDictionary resource_counts::to_map() const {
  return CountsDictionary(gateCounts);
}

void resource_counts::addQubit() { numQubits++; }

} // namespace cudaq
