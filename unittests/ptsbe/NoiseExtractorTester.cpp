/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/NoiseModel.h"
#include "common/Trace.h"
#include "cudaq/ptsbe/NoiseExtractor.h"
#include "cudaq/qis/execution_manager.h"
#include <gtest/gtest.h>

using namespace cudaq::ptsbe;

cudaq::Trace createSimpleCircuit() {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {cudaq::QuditInfo(2, 0)},
                          {cudaq::QuditInfo(2, 1)});
  trace.appendInstruction("mz", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("mz", {}, {}, {cudaq::QuditInfo(2, 1)});
  return trace;
}

cudaq::Trace createParameterizedCircuit() {
  cudaq::Trace trace;
  trace.appendInstruction("rx", {0.5}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("ry", {1.0}, {}, {cudaq::QuditInfo(2, 1)});
  trace.appendInstruction("z", {}, {cudaq::QuditInfo(2, 0)},
                          {cudaq::QuditInfo(2, 1)});
  return trace;
}

TEST(NoiseExtractorTest, EmptyCircuit) {
  cudaq::Trace empty_trace;
  cudaq::noise_model noise_model;

  auto result = extractNoiseSites(empty_trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 0);
  EXPECT_EQ(result.total_instructions, 0);
  EXPECT_EQ(result.noisy_instructions, 0);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, NoNoiseModel) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 0);
  EXPECT_EQ(result.total_instructions, 4);
  EXPECT_EQ(result.noisy_instructions, 0);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, SingleQubitDepolarization) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_EQ(result.total_instructions, 4);
  EXPECT_EQ(result.noisy_instructions, 1);
  EXPECT_TRUE(result.all_unitary_mixtures);

  const auto &np = result.noise_sites[0];
  EXPECT_EQ(np.circuit_location, 0);
  EXPECT_EQ(np.op_name, "h");
  EXPECT_EQ(np.qubits.size(), 1);
  EXPECT_EQ(np.qubits[0], 0);
  EXPECT_EQ(np.channel.size(), 4);
  EXPECT_EQ(np.channel.probabilities.size(), 4);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, TwoQubitDepolarization) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.05));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_EQ(result.noisy_instructions, 1);
  EXPECT_TRUE(result.all_unitary_mixtures);

  const auto &np = result.noise_sites[0];
  EXPECT_EQ(np.circuit_location, 1);
  EXPECT_EQ(np.op_name, "x");
  EXPECT_GE(np.qubits.size(), 1);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, MultipleNoiseSites) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.02));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 2);
  EXPECT_EQ(result.noisy_instructions, 2);
  EXPECT_TRUE(result.all_unitary_mixtures);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 0);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 1);
}

TEST(NoiseExtractorTest, BitFlipChannelIsUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.1));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
  EXPECT_TRUE(result.noise_sites[0].channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, PhaseFlipChannelIsUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::phase_flip_channel(0.05));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
  EXPECT_TRUE(result.noise_sites[0].channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, AmplitudeDampingIsNotUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::amplitude_damping_channel(0.1));

  EXPECT_THROW(
      {
        auto result = extractNoiseSites(trace, noise_model, true);
        (void)result;
      },
      std::invalid_argument);
}

TEST(NoiseExtractorTest, AmplitudeDampingWithValidationDisabled) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::amplitude_damping_channel(0.1));

  EXPECT_THROW(
      {
        auto result = extractNoiseSites(trace, noise_model, false);
        (void)result;
      },
      std::invalid_argument);
}

TEST(NoiseExtractorTest, MixedUnitaryAndNonUnitary) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::amplitude_damping_channel(0.1));
  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.1));

  EXPECT_THROW(
      {
        auto result = extractNoiseSites(trace, noise_model, true);
        (void)result;
      },
      std::invalid_argument);
  EXPECT_THROW(
      {
        auto result = extractNoiseSites(trace, noise_model, false);
        (void)result;
      },
      std::invalid_argument);
}

TEST(NoiseExtractorTest, PreservesInstructionOrder) {
  cudaq::Trace trace;

  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("y", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("z", {}, {}, {cudaq::QuditInfo(2, 0)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("y", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("z", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 4);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 0);
  EXPECT_EQ(result.noise_sites[0].op_name, "h");

  EXPECT_EQ(result.noise_sites[1].circuit_location, 1);
  EXPECT_EQ(result.noise_sites[1].op_name, "x");

  EXPECT_EQ(result.noise_sites[2].circuit_location, 2);
  EXPECT_EQ(result.noise_sites[2].op_name, "y");

  EXPECT_EQ(result.noise_sites[3].circuit_location, 3);
  EXPECT_EQ(result.noise_sites[3].op_name, "z");
}

TEST(NoiseExtractorTest, HandlesGapsInNoisyInstructions) {
  cudaq::Trace trace;

  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("y", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("z", {}, {}, {cudaq::QuditInfo(2, 0)});

  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("z", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 2);
  EXPECT_EQ(result.total_instructions, 4);
  EXPECT_EQ(result.noisy_instructions, 2);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 0);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 3);
}

TEST(NoiseExtractorTest, GracefulValidation_ValidChannels) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model, false);

  ASSERT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, ProbabilitiesSumToOne) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);

  const auto &probs = result.noise_sites[0].channel.probabilities;
  double sum = 0.0;
  for (auto p : probs) {
    sum += p;
  }

  EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(NoiseExtractorTest, AllProbabilitiesNonNegative) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.05));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);

  for (auto p : result.noise_sites[0].channel.probabilities) {
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
  }
}

TEST(NoiseExtractorTest, SingleQubitGateTracksQubit) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);
  EXPECT_EQ(result.noise_sites[0].qubits.size(), 1);
  EXPECT_EQ(result.noise_sites[0].qubits[0], 0);
}

TEST(NoiseExtractorTest, TwoQubitGateTracksQubits) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);

  const auto &qubits = result.noise_sites[0].qubits;
  EXPECT_GE(qubits.size(), 1);

  bool has_qubit_0 = std::find(qubits.begin(), qubits.end(), 0) != qubits.end();
  bool has_qubit_1 = std::find(qubits.begin(), qubits.end(), 1) != qubits.end();

  EXPECT_TRUE(has_qubit_0 || has_qubit_1);
}

TEST(NoiseExtractorTest, ToleranceParameter) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result1 = extractNoiseSites(trace, noise_model, true);
  auto result2 = extractNoiseSites(trace, noise_model, true);

  EXPECT_EQ(result1.noise_sites.size(), result2.noise_sites.size());
  EXPECT_TRUE(result1.all_unitary_mixtures);
  EXPECT_TRUE(result2.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, LargeCircuit) {
  cudaq::Trace trace;

  for (int i = 0; i < 100; ++i) {
    trace.appendInstruction(
        "h", {}, {}, {cudaq::QuditInfo(2, static_cast<std::size_t>(i % 10))});
  }

  cudaq::noise_model noise_model;

  for (std::size_t q = 0; q < 10; ++q) {
    noise_model.add_channel("h", {q}, cudaq::depolarization_channel(0.01));
  }

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.total_instructions, 100);
  EXPECT_EQ(result.noise_sites.size(), 100);
  EXPECT_EQ(result.noisy_instructions, 100);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, KrausConversion) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.1));

  auto result = extractNoiseSites(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);

  const auto &kraus_ops = result.noise_sites[0].channel.get_ops();

  EXPECT_EQ(kraus_ops.size(), 2);

  for (const auto &op : kraus_ops) {
    EXPECT_EQ(op.data.size(), 4);
  }

  for (const auto &op : kraus_ops) {
    for (const auto &elem : op.data) {
      EXPECT_TRUE(std::isfinite(elem.real()));
      EXPECT_TRUE(std::isfinite(elem.imag()));
    }
  }
}

TEST(NoiseExtractorTest, ValidationErrorMessages) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::amplitude_damping_channel(0.1));

  try {
    auto result = extractNoiseSites(trace, noise_model, true);
    (void)result;
    FAIL() << "Should have thrown std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    std::string error_msg(e.what());
    EXPECT_NE(error_msg.find("h"), std::string::npos);
    EXPECT_NE(error_msg.find("0"), std::string::npos);
    EXPECT_NE(error_msg.find("unitary mixture"), std::string::npos);
  }
}

TEST(NoiseExtractorTest, Integration_MultipleChannelsSameInstruction) {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.01));
  noise_model.add_channel("h", {0}, cudaq::phase_flip_channel(0.01));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_GE(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, Integration_DifferentNoiseTypes) {
  cudaq::Trace trace;

  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 1)});
  trace.appendInstruction("y", {}, {}, {cudaq::QuditInfo(2, 2)});
  trace.appendInstruction("z", {}, {}, {cudaq::QuditInfo(2, 3)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {1}, cudaq::bit_flip_channel(0.02));
  noise_model.add_channel("y", {2}, cudaq::phase_flip_channel(0.015));
  noise_model.add_channel("z", {3}, cudaq::depolarization_channel(0.025));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 4);
  EXPECT_EQ(result.noisy_instructions, 4);
  EXPECT_TRUE(result.all_unitary_mixtures);

  for (std::size_t i = 0; i < result.noise_sites.size(); ++i) {
    EXPECT_EQ(result.noise_sites[i].circuit_location, i);
  }
}

TEST(NoiseExtractorTest, Integration_ComplexCircuitStructure) {
  cudaq::Trace trace;

  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 1)});
  trace.appendInstruction("x", {}, {cudaq::QuditInfo(2, 0)},
                          {cudaq::QuditInfo(2, 1)});
  trace.appendInstruction("y", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("z", {}, {cudaq::QuditInfo(2, 1)},
                          {cudaq::QuditInfo(2, 2)});
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 2)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.02));
  noise_model.add_channel("h", {2}, cudaq::bit_flip_channel(0.015));

  auto result = extractNoiseSites(trace, noise_model);

  EXPECT_EQ(result.total_instructions, 6);
  EXPECT_EQ(result.noise_sites.size(), 3);
  EXPECT_TRUE(result.all_unitary_mixtures);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 0);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 2);
  EXPECT_EQ(result.noise_sites[2].circuit_location, 5);
}
