/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Resources.h"
#include "common/FmtCore.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace cudaq {
std::size_t
Resources::InstructionHash::operator()(const Instruction &instruction) const {
  std::size_t seed = 0;

  seed ^= std::hash<std::string>()(instruction.name) +
          std::hash<std::size_t>()(instruction.nControls) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2);
  return seed;
}

bool Resources::Instruction::operator==(const Instruction &other) const {
  return name == other.name && nControls == other.nControls;
}

std::size_t Resources::count(const Instruction &instruction) const {
  auto iter = instructions.find(instruction);
  if (iter == instructions.end())
    return 0;

  return iter->second;
}

std::size_t Resources::count_controls(const std::string &name,
                                      std::size_t nControls) const {
  auto iter = instructions.find({name, nControls});
  if (iter == instructions.end())
    return 0;

  return iter->second;
}

std::size_t Resources::count(const std::string &name) const {
  std::size_t result = 0;
  for (auto &[instruction, count] : instructions)
    if (instruction.name == name)
      result += count;

  return result;
}

std::size_t Resources::count() const { return totalGates; }

void Resources::appendInstruction(const std::string &name,
                                  std::size_t nControls, std::size_t count) {
  Instruction instruction(name, nControls);
  auto iter = instructions.find(instruction);
  if (iter != instructions.end()) {
    iter->second += count;
  } else {
    instructions.insert({instruction, count});
  }

  totalGates += count;
}

void Resources::dump(std::ostream &os) const {
  os << "Total # of gates: " << totalGates;
  os << ", total # of qubits: " << numQubits;
  os << "\n";
  os << "{ ";
  os << "\n  ";
  std::size_t counter = 0;
  for (auto &result : instructions) {
    std::string gatestr(result.first.nControls, 'c');
    gatestr += result.first.name;
    os << gatestr << " :  " << result.second;
    bool isLast = counter == instructions.size() - 1;
    counter++;
    os << "\n" << (!isLast ? "  " : "");
  }
  os << "}\n";
}

void Resources::dump() const { dump(std::cout); }

void Resources::clear() {
  instructions.clear();
  numQubits = 0;
  totalGates = 0;
}

void Resources::addQubit() { numQubits++; }

std::unordered_map<std::string, std::size_t> Resources::gateCounts() const {
  std::unordered_map<std::string, std::size_t> gateCounts;
  for (auto &result : instructions) {
    std::string gatestr(result.first.nControls, 'c');
    gatestr += result.first.name;
    gateCounts[gatestr] = result.second;
  }
  return gateCounts;
}
} // namespace cudaq
