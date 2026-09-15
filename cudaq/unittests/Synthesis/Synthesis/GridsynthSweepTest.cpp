/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Cost regression sweep. The rest of the suite checks properties, which keep
// holding after an optimization quietly changes the circuit. This pins the
// answers. GridsynthSweepData.h records the error, T-count and gate count for
// every (theta, epsilon, seed) below.
//
// After an intentional change, print a fresh table and paste it between the
// R"CSV( and )CSV" markers in that header, then read the diff -- the T-count
// must not move and the gate count must not grow. The table goes to stderr and
// gtest's own output to stdout, so the redirect captures only the table
//
//   ./synthesis_gridsynth_sweep_test --gtest_also_run_disabled_tests \
//       --gtest_filter='*PrintSweepData*' 2> table.csv

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "GridsynthSweepData.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/Support/LogicalResult.h"

namespace {

using cudaq::synth::Circuit;
using cudaq::synth::Real;
using cudaq::synth::ScopedDefaultPrecision;

// pi/4 and 3pi/4 sit on the line the ODGP bound refinement is sensitive to,
// and pi/16 through pi/128 approach zero, where the epsilon region degenerates.
const std::pair<int, int> kPiMultiples[] = {
    {1, 3},  {1, 4},  {1, 5},  {1, 6},   {1, 7},   {1, 8},
    {1, 16}, {1, 32}, {1, 64}, {1, 128}, {2, 3},   {2, 5},
    {3, 4},  {3, 8},  {5, 8},  {7, 9},   {11, 13}, {15, 16}};

const char *kPlainAngles[] = {"0.5", "1.0", "2.0",
                              "0.1234567890123456789012345678901234567890"};

const char *kEpsilons[] = {"1e-2", "1e-3", "1e-4", "1e-5",
                           "1e-6", "1e-7", "1e-8"};

const uint64_t kSeeds[] = {1, 2, 3, 4, 5, 6, 7};

std::vector<std::string> sweep_angles() {
  std::vector<std::string> angles;
  {
    ScopedDefaultPrecision high(512);
    for (const auto &[p, q] : kPiMultiples)
      angles.push_back((Real::pi() * Real(p) / Real(q)).to_string(40));
  }
  for (const char *angle : kPlainAngles)
    angles.push_back(angle);
  return angles;
}

struct Cost {
  std::string error;
  int t_count = 0;
  size_t gate_count = 0;
};

llvm::FailureOr<Cost> synthesize(const std::string &theta,
                                 const std::string &epsilon, uint64_t seed) {
  ScopedDefaultPrecision prec(
      cudaq::synth::details::required_precision(Real(epsilon)));

  cudaq::synth::GridsynthOptions options;
  options.seed = seed;
  llvm::FailureOr<Circuit> circuit =
      cudaq::synth::gridsynth(Real(theta), Real(epsilon), options);
  if (llvm::failed(circuit))
    return llvm::failure();

  return Cost{cudaq::synth::rz_gate_sequence_error(theta, *circuit),
              circuit->t_count(), circuit->size()};
}

struct Row {
  std::string theta;
  std::string epsilon;
  uint64_t seed = 0;
  Cost cost;
};

std::vector<Row> recorded_rows() {
  std::vector<Row> rows;
  std::istringstream data(cudaq::synth::test::kSweepData);
  std::string line;
  while (std::getline(data, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream fields(line);
    std::string seed, t_count, gate_count;
    Row row;
    std::getline(fields, row.theta, ',');
    std::getline(fields, row.epsilon, ',');
    std::getline(fields, seed, ',');
    std::getline(fields, row.cost.error, ',');
    std::getline(fields, t_count, ',');
    std::getline(fields, gate_count, ',');
    row.seed = std::stoull(seed);
    row.cost.t_count = std::stoi(t_count);
    row.cost.gate_count = std::stoull(gate_count);
    rows.push_back(row);
  }
  return rows;
}

TEST(GridsynthSweepTest, CostsMatchTheRecordedFixture) {
  const std::vector<Row> rows = recorded_rows();
  ASSERT_FALSE(rows.empty()) << "GridsynthSweepData.h carries no cases";

  for (const Row &row : rows) {
    SCOPED_TRACE("theta=" + row.theta + " epsilon=" + row.epsilon +
                 " seed=" + std::to_string(row.seed));

    llvm::FailureOr<Cost> cost = synthesize(row.theta, row.epsilon, row.seed);
    if (llvm::failed(cost)) {
      ADD_FAILURE() << "synthesis failed";
      continue;
    }

    EXPECT_LE(Real(cost->error), Real(row.epsilon))
        << "error " << cost->error << " exceeds epsilon";
    EXPECT_EQ(cost->t_count, row.cost.t_count) << "T-count moved";
    EXPECT_LE(cost->gate_count, row.cost.gate_count) << "gate count grew";
  }
}

// Prints a fresh table to stderr.
TEST(GridsynthSweepTest, DISABLED_PrintSweepData) {
  for (const std::string &theta : sweep_angles())
    for (const char *epsilon : kEpsilons)
      for (uint64_t seed : kSeeds) {
        llvm::FailureOr<Cost> cost = synthesize(theta, epsilon, seed);
        ASSERT_TRUE(llvm::succeeded(cost))
            << "synthesis failed for theta=" << theta << " epsilon=" << epsilon
            << " seed=" << seed;
        std::cerr << theta << ',' << epsilon << ',' << seed << ','
                  << Real(cost->error).to_string(12) << ',' << cost->t_count
                  << ',' << cost->gate_count << '\n';
      }
}

} // namespace
