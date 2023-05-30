/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
  std::size_t cHash = 0;
  for (auto &c : instruction.controls)
    cHash ^= std::hash<std::size_t>()(c);

  seed ^= std::hash<std::string>()(instruction.name) + cHash +
          std::hash<std::size_t>()(instruction.target) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2);
  return seed;
}

bool Resources::Instruction::operator==(const Instruction &other) const {
  return name == other.name && controls == other.controls &&
         target == other.target;
}

std::size_t Resources::count(const Instruction &instruction) const {
  auto iter = instructions.find(instruction);
  if (iter == instructions.end())
    return 0;

  return iter->second;
}

std::size_t Resources::count_controls(const std::string &name,
                                      std::size_t nControls) const {
  std::size_t result = 0;
  for (auto &[instruction, count] : instructions)
    if (instruction.name == name && instruction.controls.size() == nControls)
      result++;

  return result;
}

std::size_t Resources::count(const std::string &name) const {
  std::size_t result = 0;
  for (auto &[instruction, count] : instructions)
    if (instruction.name == name)
      result++;

  return result;
}

std::size_t Resources::count() const {
  std::size_t total = 0;
  for (const auto &[instruction, count] : instructions)
    total += count;
  return total;
}

void Resources::appendInstruction(const Resources::Instruction &instruction) {
  auto iter = instructions.find(instruction);
  if (iter == instructions.end())
    instructions.insert(std::make_pair(instruction, 1));
  else
    iter->second++;
}

void Resources::dump(std::ostream &os) const {
  os << "Resources Estimation Result:\n";
  size_t totalNumberGates = 0, totalCtrlOperations = 0, numQubits = 0;
  std::stringstream stream;
  const size_t columns = 2;
  const size_t columnWidth = 8;
  const auto totalWidth = columns * columnWidth + 6;
  stream << std::string(totalWidth, '-') << "\n";
  stream << "| " << std::left << std::setw(8) << "Operation"
         << " |";
  stream << std::left << std::setw(8) << "Count"

         << " |\n";
  stream << std::string(totalWidth, '-') << "\n";
  const auto writeRow = [&](const Instruction &inst, int count) {
    if (!inst.controls.empty()) {
      stream << "| " << std::setw(8)
             << fmt::format("{}[{}] {}", inst.name,
                            fmt::join(inst.controls, ", "), inst.target)
             << " | ";
      auto maxQubitIdx =
          *std::max_element(inst.controls.begin(), inst.controls.end()) + 1;
      if (maxQubitIdx > numQubits) {
        numQubits = maxQubitIdx;
      }
    } else {
      // we don't have controls
      stream << "| " << std::setw(8)
             << fmt::format("{}({})", inst.name, inst.target) << " | ";
    }

    if (inst.target + 1 > numQubits) {
      numQubits = inst.target + 1;
    }
    stream << std::setw(8) << count << " |\n";
  };

  // Print each row, and count total number of instructions, and
  // count any 2 qubit control operations.
  for (const auto &[instruction, count] : instructions) {
    writeRow(instruction, count);
    totalNumberGates += count;
    if (!instruction.controls.empty()) {
      totalCtrlOperations += count;
    }
  }

  stream << std::string(totalWidth, '-') << "\n";
  os << "Number of qubits required: " << numQubits << "\n";
  os << "Total Operations: " << totalNumberGates << "\n";
  os << "Total Control Operations: " << totalCtrlOperations << "\n";
  os << "Operation Count Report: \n";
  os << stream.str();
}
void Resources::dump() const { dump(std::cout); }
} // namespace cudaq