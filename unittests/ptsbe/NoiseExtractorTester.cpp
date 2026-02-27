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
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/qis/execution_manager.h"
#include <gtest/gtest.h>

using namespace cudaq::ptsbe;

/// Build PTSBE trace then extract noise sites (the two-step pipeline).
static NoiseExtractionResult buildAndExtract(const cudaq::Trace &trace,
                                             const cudaq::noise_model &noise,
                                             bool validate = true) {
  auto ptsbe = buildPTSBETrace(trace, noise);
  return extractNoiseSites(ptsbe, validate);
}

cudaq::Trace createSimpleCircuit() {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {cudaq::QuditInfo(2, 0)},
                          {cudaq::QuditInfo(2, 1)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 0)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 1)});
  return trace;
}

TEST(NoiseExtractorTest, EmptyCircuit) {
  cudaq::Trace empty_trace;
  cudaq::noise_model noise_model;

  auto result = buildAndExtract(empty_trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 0);
  EXPECT_EQ(result.total_instructions, 0);
  EXPECT_EQ(result.noisy_instructions, 0);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, NoNoiseModel) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 0);
  // Gate(h), Gate(x), Measurement(mz q0), Measurement(mz q1)
  EXPECT_EQ(result.total_instructions, 4);
  EXPECT_EQ(result.noisy_instructions, 0);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, SingleQubitDepolarization) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  // Gate(h), Noise(depol), Gate(x), Meas(q0), Meas(q1)
  EXPECT_EQ(result.total_instructions, 5);
  EXPECT_EQ(result.noisy_instructions, 1);
  EXPECT_TRUE(result.all_unitary_mixtures);

  const auto &np = result.noise_sites[0];
  EXPECT_EQ(np.circuit_location, 1);
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

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_EQ(result.noisy_instructions, 1);
  EXPECT_TRUE(result.all_unitary_mixtures);

  const auto &np = result.noise_sites[0];
  // Gate(h), Gate(x), Noise(depol2), Meas, Meas
  EXPECT_EQ(np.circuit_location, 2);
  EXPECT_GE(np.qubits.size(), 1);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, MultipleNoiseSites) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.02));

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 2);
  EXPECT_EQ(result.noisy_instructions, 2);
  EXPECT_TRUE(result.all_unitary_mixtures);

  // Gate(h), Noise(depol), Gate(x), Noise(depol2), Meas, Meas
  EXPECT_EQ(result.noise_sites[0].circuit_location, 1);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 3);
}

TEST(NoiseExtractorTest, BitFlipChannelIsUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.1));

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
  EXPECT_TRUE(result.noise_sites[0].channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, PhaseFlipChannelIsUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::phase_flip_channel(0.05));

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
  EXPECT_TRUE(result.noise_sites[0].channel.is_unitary_mixture());
}

TEST(NoiseExtractorTest, AmplitudeDampingIsNotUnitaryMixture) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::amplitude_damping_channel(0.1));

  try {
    auto result = buildAndExtract(trace, noise_model, true);
    (void)result;
    FAIL() << "Expected an exception for non-unitary-mixture channel";
  } catch (...) {
  }
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

  auto result = buildAndExtract(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 4);

  // Noise entries are interleaved: G,N,G,N,G,N,G,N -> locations 1,3,5,7
  EXPECT_EQ(result.noise_sites[0].circuit_location, 1);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 3);
  EXPECT_EQ(result.noise_sites[2].circuit_location, 5);
  EXPECT_EQ(result.noise_sites[3].circuit_location, 7);

  // Verify strictly increasing order
  for (std::size_t i = 1; i < result.noise_sites.size(); ++i)
    EXPECT_GT(result.noise_sites[i].circuit_location,
              result.noise_sites[i - 1].circuit_location);
}

TEST(NoiseExtractorTest, HandlesGapsInNoisyInstructions) {
  cudaq::Trace trace;

  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("y", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("z", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 0)});

  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("z", {0}, cudaq::depolarization_channel(0.01));

  auto result = buildAndExtract(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 2);
  // Gate(h), Noise, Gate(x), Gate(y), Gate(z), Noise, Measurement = 7
  EXPECT_EQ(result.total_instructions, 7);
  EXPECT_EQ(result.noisy_instructions, 2);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 1);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 5);
}

TEST(NoiseExtractorTest, GracefulValidation_ValidChannels) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = buildAndExtract(trace, noise_model, false);

  ASSERT_EQ(result.noise_sites.size(), 1);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, SingleQubitGateTracksQubit) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  auto result = buildAndExtract(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);
  EXPECT_EQ(result.noise_sites[0].qubits.size(), 1);
  EXPECT_EQ(result.noise_sites[0].qubits[0], 0);
}

TEST(NoiseExtractorTest, TwoQubitGateTracksQubits) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;

  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.01));

  auto result = buildAndExtract(trace, noise_model);

  ASSERT_EQ(result.noise_sites.size(), 1);

  const auto &qubits = result.noise_sites[0].qubits;
  EXPECT_GE(qubits.size(), 1);

  bool has_qubit_0 = std::find(qubits.begin(), qubits.end(), 0) != qubits.end();
  bool has_qubit_1 = std::find(qubits.begin(), qubits.end(), 1) != qubits.end();

  EXPECT_TRUE(has_qubit_0 || has_qubit_1);
}

TEST(NoiseExtractorTest, LargeCircuit) {
  constexpr std::size_t NUM_QUBITS = 10;
  constexpr std::size_t NUM_GATES = 100;

  cudaq::Trace trace;
  for (std::size_t i = 0; i < NUM_GATES; ++i) {
    trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, i % NUM_QUBITS)});
  }
  for (std::size_t q = 0; q < NUM_QUBITS; ++q) {
    trace.appendMeasurement("mz", {cudaq::QuditInfo(2, q)});
  }

  cudaq::noise_model noise_model;
  for (std::size_t q = 0; q < NUM_QUBITS; ++q) {
    noise_model.add_channel("h", {q}, cudaq::depolarization_channel(0.01));
  }

  auto result = buildAndExtract(trace, noise_model);

  // 100 gates + 100 noise + 10 measurements = 210
  EXPECT_EQ(result.total_instructions, NUM_GATES * 2 + NUM_QUBITS);
  EXPECT_EQ(result.noise_sites.size(), NUM_GATES);
  EXPECT_EQ(result.noisy_instructions, NUM_GATES);
  EXPECT_TRUE(result.all_unitary_mixtures);
}

TEST(NoiseExtractorTest, KrausConversion) {
  auto trace = createSimpleCircuit();
  cudaq::noise_model noise_model;
  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.1));

  auto result = buildAndExtract(trace, noise_model);

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
    auto result = buildAndExtract(trace, noise_model, true);
    (void)result;
    FAIL() << "Should have thrown for non-unitary-mixture channel";
  } catch (...) {
  }
}

TEST(NoiseExtractorTest, Integration_MultipleChannelsSameInstruction) {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::bit_flip_channel(0.01));
  noise_model.add_channel("h", {0}, cudaq::phase_flip_channel(0.01));

  auto result = buildAndExtract(trace, noise_model);

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

  auto result = buildAndExtract(trace, noise_model);

  EXPECT_EQ(result.noise_sites.size(), 4);
  EXPECT_EQ(result.noisy_instructions, 4);
  EXPECT_TRUE(result.all_unitary_mixtures);

  // Interleaved: G,N,G,N,G,N,G,N -> locations 1,3,5,7
  for (std::size_t i = 1; i < result.noise_sites.size(); ++i)
    EXPECT_GT(result.noise_sites[i].circuit_location,
              result.noise_sites[i - 1].circuit_location);
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
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 0)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 1)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 2)});

  cudaq::noise_model noise_model;

  noise_model.add_channel("h", {0}, cudaq::depolarization_channel(0.01));
  noise_model.add_channel("x", {0, 1}, cudaq::depolarization2(0.02));
  noise_model.add_channel("h", {2}, cudaq::bit_flip_channel(0.015));

  auto result = buildAndExtract(trace, noise_model);

  // Gate(h,q0), Noise, Gate(h,q1), Gate(x), Noise, Gate(y), Gate(z),
  // Gate(h,q2), Noise, Mz(q0), Mz(q1), Mz(q2) = 12
  EXPECT_EQ(result.total_instructions, 12);
  EXPECT_EQ(result.noise_sites.size(), 3);
  EXPECT_TRUE(result.all_unitary_mixtures);

  EXPECT_EQ(result.noise_sites[0].circuit_location, 1);
  EXPECT_EQ(result.noise_sites[1].circuit_location, 4);
  EXPECT_EQ(result.noise_sites[2].circuit_location, 8);
}

TEST(NoiseExtractorTest, ImplicitMeasurementPerQubitNoise) {
  cudaq::Trace trace;
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {cudaq::QuditInfo(2, 1)});

  cudaq::noise_model noise_model;
  noise_model.add_channel("mz", {0}, cudaq::bit_flip_channel(0.1));
  noise_model.add_channel("mz", {1}, cudaq::bit_flip_channel(0.2));

  auto ptsbe = buildPTSBETrace(trace, noise_model);

  std::size_t measCount = 0;
  std::size_t noiseCount = 0;
  for (const auto &inst : ptsbe) {
    if (inst.type == TraceInstructionType::Measurement)
      ++measCount;
    if (inst.type == TraceInstructionType::Noise)
      ++noiseCount;
  }
  EXPECT_EQ(measCount, 2);
  EXPECT_EQ(noiseCount, 2);

  auto result = extractNoiseSites(ptsbe);
  EXPECT_EQ(result.noise_sites.size(), 2);
  EXPECT_NE(result.noise_sites[0].qubits, result.noise_sites[1].qubits);
}
