/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudaq::synth::Circuit;
using cudaq::synth::DOmegaUnitary;
using cudaq::synth::Gate;

TEST(CircuitNormalizationTest, RepresentativeSequences) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"I", "I"}, {"SSSS", "I"}, {"TT", "S"}, {"TST", "SS"}, {"SXS", "XWW"}};

  for (const auto &[inputString, expectedString] : cases) {
    auto input = Circuit::from_string(inputString);
    ASSERT_TRUE(llvm::succeeded(input));
    auto normalized = input->normalized();

    EXPECT_EQ(normalized.to_string(), expectedString) << inputString;
    EXPECT_EQ(DOmegaUnitary::from_gates(*input),
              DOmegaUnitary::from_gates(normalized))
        << inputString;
    EXPECT_EQ(normalized.normalized(), normalized) << inputString;
    EXPECT_LE(normalized.t_count(), input->t_count()) << inputString;
  }
}

TEST(CircuitNormalizationTest, PreservesBoundedShortWords) {
  std::vector<Circuit> words{Circuit{}};
  const std::vector<Gate> alphabet = {Gate::H, Gate::S, Gate::T, Gate::X,
                                      Gate::W};

  for (int length = 1; length <= 4; ++length) {
    std::vector<Circuit> next;
    next.reserve(words.size() * alphabet.size());
    for (const auto &prefix : words)
      for (Gate gate : alphabet) {
        Circuit word = prefix;
        word.push_back(gate);
        auto normalized = word.normalized();
        EXPECT_EQ(DOmegaUnitary::from_gates(word),
                  DOmegaUnitary::from_gates(normalized))
            << word.to_string();
        EXPECT_EQ(normalized.normalized(), normalized) << word.to_string();
        EXPECT_LE(normalized.t_count(), word.t_count()) << word.to_string();
        next.push_back(std::move(word));
      }
    words = std::move(next);
  }
}

} // namespace
