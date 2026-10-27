/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "CustabilizerCircuitSimulator.cpp"
#include <random>
#include <set>

namespace {

// Check the restriction on the serialized circuit that cuStabilizer receives.
// Reparse it so missing TICK boundaries cannot hide behind in-memory
// operations.
void expectDisjointTargets(const stim::Circuit &recorded) {
  const auto text = recorded.str();
  SCOPED_TRACE(text);
  const stim::Circuit reparsed(text);
  reparsed.for_each_operation([&](const stim::CircuitInstruction &instruction) {
    const auto flags = stim::GATE_DATA[instruction.gate_type].flags;
    if (!(flags & (stim::GATE_IS_UNITARY | stim::GATE_IS_NOISY |
                   stim::GATE_IS_RESET | stim::GATE_PRODUCES_RESULTS)))
      return;
    std::set<std::uint32_t> qubits;
    for (const auto &target : instruction.targets) {
      if (target.has_qubit_value())
        EXPECT_TRUE(qubits.insert(target.qubit_value()).second)
            << "Qubit " << target.qubit_value()
            << " occurs twice in one instruction";
    }
  });
}

} // namespace

CUDAQ_TEST(CustabilizerResults, PackedConversionMatchesScalarReference) {
  // Cover partial bytes, words and tiles, skipped measurement prefixes, and
  // strides larger than the requested shot count. Random padding must not
  // appear in the output.
  std::mt19937_64 random(1997);
  for (std::size_t width :
       {0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 801, 5185}) {
    for (std::size_t shots : {1, 31, 32, 33, 63, 64, 65, 129, 257}) {
      for (std::size_t extraWords : {0, 2}) {
        const auto words = (shots + 31) / 32 + extraWords;
        for (std::size_t prefix : {0, 3}) {
          SCOPED_TRACE(::testing::Message()
                       << "width=" << width << " shots=" << shots
                       << " words=" << words << " prefix=" << prefix);
          std::vector<bool> reference(width + prefix);
          for (std::size_t m = 0; m < reference.size(); ++m)
            reference[m] = random() & 1;
          std::vector<std::uint32_t> packed(reference.size() * words);
          for (auto &word : packed)
            word = random();
          cudaq::ExecutionResult expected;
          for (std::size_t s = 0; s < shots; ++s) {
            std::string bits;
            for (std::size_t m = prefix; m < reference.size(); ++m) {
              const auto flip = (packed[m * words + s / 32] >> (s % 32)) & 1;
              bits.push_back((flip ^ reference[m]) ? '1' : '0');
            }
            ++expected.counts[bits];
            expected.sequentialData.push_back(std::move(bits));
          }
          for (bool sequential : {false, true}) {
            const auto actual = nvqir::makeSampleResult(
                packed, words, reference, prefix, shots, sequential);
            EXPECT_EQ(actual.counts, expected.counts);
            if (sequential)
              EXPECT_EQ(actual.sequentialData, expected.sequentialData);
            else
              EXPECT_TRUE(actual.sequentialData.empty());
          }
        }
      }
    }
  }
}

CUDAQ_TEST(FrameRunner, SeparatesOverlapsAcrossAppendCalls) {
  nvqir::CustabilizerFrameRunner runner(4);
  for (const auto &text : {"CX 0 1", "CX 0 2", "CX 1 2", "CX 1 2"})
    runner.apply(stim::Circuit(text));

  const stim::Circuit expected(
      "CX 0 1\nTICK\nCX 0 2\nTICK\nCX 1 2\nTICK\nCX 1 2");
  EXPECT_EQ(stim::Circuit(runner.circuitCache.str()), expected);
  expectDisjointTargets(runner.circuitCache);
}

CUDAQ_TEST(FrameRunner, KeepsDisjointOperationsFused) {
  nvqir::CustabilizerFrameRunner runner(8);
  runner.apply(stim::Circuit("CX 0 1"));
  runner.apply(stim::Circuit("CX 2 3"));
  runner.apply(stim::Circuit("CX 4 5 6 7"));

  const stim::Circuit reparsed(runner.circuitCache.str());
  ASSERT_EQ(reparsed.operations.size(), 1);
  EXPECT_EQ(reparsed.operations.front().gate_type, stim::GateType::CX);
  EXPECT_EQ(reparsed.operations.front().targets.size(), 8);
  expectDisjointTargets(runner.circuitCache);
}

CUDAQ_TEST(FrameRunner, SplitsAlreadyFusedInstructions) {
  nvqir::CustabilizerFrameRunner runner(4);
  for (const auto &text : {"CX 0 1 0 2 0 3", "CX 0 3 1 3 2 3", "CX 0 1 1 2 2 3",
                           "CX 0 1 0 1", "H 0 0 1 2 1", "R 0 0 1 1", "M 0 0 1",
                           "X_ERROR(1) 0 0 1 1", "DEPOLARIZE2(0.1) 0 1 1 2"}) {
    SCOPED_TRACE(text);
    runner.clear();
    const stim::Circuit input(text);
    ASSERT_EQ(input.operations.size(), 1);
    runner.apply(input);
    EXPECT_GT(runner.circuitCache.count_ticks(), 0);
    EXPECT_EQ(runner.circuitCache.count_measurements(),
              input.count_measurements());
    expectDisjointTargets(runner.circuitCache);
  }
}

CUDAQ_TEST(FrameRunner, PreservesExplicitTicks) {
  nvqir::CustabilizerFrameRunner runner(3);
  runner.apply(stim::Circuit("H 0\nTICK\nH 1"));
  runner.apply(stim::Circuit("TICK\nTICK\nH 2"));

  const stim::Circuit expected("H 0\nTICK\nH 1\nTICK\nTICK\nH 2");
  EXPECT_EQ(stim::Circuit(runner.circuitCache.str()), expected);
  expectDisjointTargets(runner.circuitCache);
}

CUDAQ_TEST(FrameRunner, ClearDiscardsPreviousCircuit) {
  nvqir::CustabilizerFrameRunner runner(4);
  runner.apply(stim::Circuit("R 0\nH 0\nM 0 0\nCX 0 1"));
  runner.clear();
  EXPECT_TRUE(runner.circuitCache.operations.empty());
  EXPECT_EQ(runner.circuitCache.count_measurements(), 0);

  runner.apply(stim::Circuit("CX 0 2"));
  runner.apply(stim::Circuit("CX 1 3\nM 3"));
  const stim::Circuit expected("CX 0 2 1 3\nM 3");
  EXPECT_EQ(stim::Circuit(runner.circuitCache.str()), expected);
  expectDisjointTargets(runner.circuitCache);
}

CUDAQ_TEST(FrameRunner, GeneratedCircuitsHaveDisjointSerializedTargets) {
  // cuStabilizer forbids overlapping qubits within one instruction. Check that
  // property directly, independently of any reference conversion algorithm.
  struct Gate {
    const char *name;
    bool pairs;
    std::vector<double> args;
  };
  const std::vector<Gate> gates{
      {"H", false, {}},
      {"S", false, {}},
      {"X", false, {}},
      {"Y", false, {}},
      {"Z", false, {}},
      {"CX", true, {}},
      {"CZ", true, {}},
      {"R", false, {}},
      {"RX", false, {}},
      {"RY", false, {}},
      {"M", false, {}},
      {"MX", false, {}},
      {"MY", false, {}},
      {"X_ERROR", false, {0.25}},
      {"Z_ERROR", false, {1.0}},
      {"DEPOLARIZE1", false, {0.125}},
      {"DEPOLARIZE2", true, {0.125}},
      {"PAULI_CHANNEL_1", false, {0.125, 0.25, 0.125}},
  };
  std::mt19937 random(2026);
  for (std::size_t circuitIndex = 0; circuitIndex < 64; ++circuitIndex) {
    SCOPED_TRACE(circuitIndex);
    const std::uint32_t numQubits = 2 + random() % 5;
    nvqir::CustabilizerFrameRunner runner(numQubits);
    std::uint64_t measurements = 0;
    for (std::size_t step = 0; step < 40; ++step) {
      SCOPED_TRACE(step);
      if (step == 20) {
        runner.clear();
        measurements = 0;
      }
      const auto &gate = gates[random() % gates.size()];
      std::vector<std::uint32_t> targets;
      const auto groups = 1 + random() % 4;
      for (std::size_t i = 0; i < groups; ++i) {
        const auto first = random() % numQubits;
        targets.push_back(first);
        if (gate.pairs)
          targets.push_back((first + 1 + random() % (numQubits - 1)) %
                            numQubits);
      }
      stim::Circuit input;
      input.safe_append_u(gate.name, targets, gate.args);
      if (random() % 7 == 0)
        input.safe_append_u("TICK", {});
      measurements += input.count_measurements();
      runner.apply(input);
      EXPECT_EQ(runner.circuitCache.count_measurements(), measurements);
      expectDisjointTargets(runner.circuitCache);
    }
  }
}
