/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

void Resources::appendInstruction(const std::string &name,
                                  const std::vector<std::size_t> &controls,
                                  const std::vector<std::size_t> &targets) {
  appendInstruction(name, controls.size());

  // Collect all qubit indices touched by this gate.
  std::vector<std::size_t> allQubits;
  allQubits.insert(allQubits.end(), controls.begin(), controls.end());
  allQubits.insert(allQubits.end(), targets.begin(), targets.end());

  // Update total depth: each qubit advances to max(touched depths) + 1.
  std::size_t maxDepth = 0;
  for (auto q : allQubits)
    maxDepth = std::max(maxDepth, perQubitDepth[q]);
  std::size_t newDepth = maxDepth + 1;
  for (auto q : allQubits)
    perQubitDepth[q] = newDepth;

  // Track gate count and depth by arity (total qubit count = controls +
  // targets, distinct from nControls used in the Instruction key).
  auto arity = allQubits.size();
  gateCountByArity[arity]++;
  auto &arityDepthMap = perQubitDepthByArity[arity];
  std::size_t maxArityDepth = 0;
  for (auto q : allQubits)
    maxArityDepth = std::max(maxArityDepth, arityDepthMap[q]);
  std::size_t newArityDepth = maxArityDepth + 1;
  for (auto q : allQubits)
    arityDepthMap[q] = newArityDepth;
}

std::size_t Resources::getNumQubits() const { return numQubits; }

std::size_t Resources::getCircuitDepth() const {
  std::size_t maxDepth = 0;
  for (auto &[qubit, depth] : perQubitDepth)
    maxDepth = std::max(maxDepth, depth);
  return maxDepth;
}

std::size_t Resources::getGateCountByArity(std::size_t arity) const {
  auto it = gateCountByArity.find(arity);
  return it != gateCountByArity.end() ? it->second : 0;
}

std::size_t Resources::getDepthByArity(std::size_t arity) const {
  auto it = perQubitDepthByArity.find(arity);
  if (it == perQubitDepthByArity.end())
    return 0;
  std::size_t maxDepth = 0;
  for (auto &[qubit, depth] : it->second)
    maxDepth = std::max(maxDepth, depth);
  return maxDepth;
}

std::size_t Resources::getMultiQubitGateCount() const {
  std::size_t total = 0;
  for (auto &[arity, count] : gateCountByArity)
    if (arity >= 2)
      total += count;
  return total;
}

std::size_t Resources::getMultiQubitDepth() const {
  std::size_t maxDepth = 0;
  for (auto &[arity, depthMap] : perQubitDepthByArity)
    if (arity >= 2)
      for (auto &[qubit, depth] : depthMap)
        maxDepth = std::max(maxDepth, depth);
  return maxDepth;
}

const std::map<std::size_t, std::size_t> &
Resources::getGateCountsByArity() const {
  return gateCountByArity;
}

const std::unordered_map<std::size_t, std::size_t> &
Resources::getPerQubitDepth() const {
  return perQubitDepth;
}

void Resources::dump(std::ostream &os) const {
  os << "Total # of gates: " << totalGates;
  os << ", total # of qubits: " << numQubits;
  os << ", circuit depth: " << getCircuitDepth();
  os << ", multi-Q gate count: " << getMultiQubitGateCount();
  os << ", multi-Q depth: " << getMultiQubitDepth();
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
  perQubitDepth.clear();
  gateCountByArity.clear();
  perQubitDepthByArity.clear();
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
