/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved. *
 * *
 * This source code and the accompanying materials are made available under *
 * the terms of the Apache License 2.0 which accompanies this distribution. *
 ******************************************************************************/

#include "common/AnalogRydberg.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

using namespace cudaq;

namespace {
ahs::Program makeProgram(std::vector<std::vector<double>> sites,
                         double omega = 0.0, double phase = 0.0,
                         double detuning = 0.0) {
  ahs::Program program;
  program.setup.ahs_register.sites = sites;
  program.setup.ahs_register.filling = std::vector<int>(sites.size(), 1);
  ahs::DrivingField drive;
  drive.amplitude.time_series = ahs::TimeSeries({{omega, 0.0}, {omega, 1e-6}});
  drive.phase.time_series = ahs::TimeSeries({{phase, 0.0}, {phase, 1e-6}});
  drive.detuning.time_series =
      ahs::TimeSeries({{detuning, 0.0}, {detuning, 1e-6}});
  program.hamiltonian.drivingFields = {drive};
  return program;
}

complex_matrix matrixAt(const analog::Model &model, double time = 0.0) {
  return model.hamiltonian.to_matrix(model.dimensions, {{"t", time}});
}
} // namespace

TEST(AHSModelTest, DeviceCoefficientsAndUnits) {
  EXPECT_DOUBLE_EQ(ahs::fresnelCan.rydbergC6, 865723.02 * 1e-30);
  for (auto device : {ahs::fresnelCan, ahs::DeviceSpecification{5.42e-24}}) {
    auto matrix = matrixAt(
        ahs::makeRydbergModel(makeProgram({{0., 0.}, {5e-6, 0.}}), device));
    EXPECT_NEAR(matrix(3, 3).real(), device.rydbergC6 / std::pow(5e-6, 6),
                1e-6);
    for (auto basis : {0, 1, 2})
      EXPECT_NEAR(std::abs(matrix(basis, basis)), 0.0, 1e-8);
    auto farther = matrixAt(
        ahs::makeRydbergModel(makeProgram({{0., 0.}, {1e-5, 0.}}), device));
    EXPECT_NEAR(matrix(3, 3).real() / farther(3, 3).real(), 64.0, 1e-12);
  }
}

TEST(AHSModelTest, PhaseSignAndDetuning) {
  auto model = ahs::makeRydbergModel(
      makeProgram({{0., 0.}}, 2e6, std::numbers::pi / 2, 3e6), ahs::fresnelCan);
  auto matrix = matrixAt(model);
  EXPECT_NEAR(std::abs(matrix(0, 1) - std::complex<double>(0., -1e6)), 0.,
              1e-9);
  EXPECT_NEAR(std::abs(matrix(1, 0) - std::complex<double>(0., 1e6)), 0., 1e-9);
  EXPECT_DOUBLE_EQ(matrix(0, 0).real(), 0.0);
  EXPECT_DOUBLE_EQ(matrix(1, 1).real(), -3e6);
}

TEST(AHSModelTest, AsymmetricRegisterOrdering) {
  auto model = ahs::makeRydbergModel(
      makeProgram({{0., 0.}, {6e-6, 0.}, {0., 8e-6}}), ahs::fresnelCan);
  auto matrix = matrixAt(model);
  EXPECT_NEAR(matrix(3, 3).real(),
              ahs::fresnelCan.rydbergC6 / std::pow(6e-6, 6), 1e-7);
  EXPECT_NEAR(matrix(5, 5).real(),
              ahs::fresnelCan.rydbergC6 / std::pow(8e-6, 6), 1e-7);
  EXPECT_NEAR(matrix(6, 6).real(),
              ahs::fresnelCan.rydbergC6 / std::pow(1e-5, 6), 1e-7);
}

TEST(AHSModelTest, WaveformInterpolation) {
  auto program = makeProgram({{0., 0.}});
  auto &drive = program.hamiltonian.drivingFields.front();
  drive.amplitude.time_series = ahs::TimeSeries({{0., 0.}, {4e6, 1e-6}});
  drive.detuning.time_series = ahs::TimeSeries({{-2e6, 0.}, {2e6, 1e-6}});
  drive.phase.time_series =
      ahs::TimeSeries({{0., 0.}, {std::numbers::pi / 2, 5e-7}, {0., 1e-6}});
  auto model = ahs::makeRydbergModel(program, ahs::fresnelCan);
  EXPECT_EQ(model.times, (std::vector<double>{0., 5e-7, 1e-6}));
  auto before = matrixAt(model, 2.5e-7);
  auto after = matrixAt(model, 7.5e-7);
  EXPECT_NEAR(std::abs(before(0, 1) - 5e5), 0., 1e-9);
  EXPECT_NEAR(before(1, 1).real(), 1e6, 1e-9);
  EXPECT_NEAR(std::abs(after(0, 1) - std::complex<double>(0., -1.5e6)), 0.,
              1e-9);
}

TEST(AHSModelTest, StepTracksInteractionStrength) {
  auto model = ahs::makeRydbergModel(makeProgram({{0., 0.}, {1e-6, 0.}}),
                                     ahs::fresnelCan);
  EXPECT_LT(model.maxStep, 1e-9);
  EXPECT_LE(model.maxStep * ahs::fresnelCan.rydbergC6 / std::pow(1e-6, 6),
            0.1000000001);
}

TEST(AHSModelTest, RejectNumericallyInvalidModels) {
  const auto pair = makeProgram({{0., 0.}, {6e-6, 0.}});
  // Non-finite C6, an infinite norm bound, and too many integration steps.
  for (auto c6 : {std::numeric_limits<double>::quiet_NaN(), 1e300, 1.0})
    EXPECT_THROW(ahs::makeRydbergModel(pair, ahs::DeviceSpecification{c6}),
                 std::invalid_argument);
  // Coincident atoms, and 0.1 um spacing that needs ~1e13 steps.
  for (auto spacing : {1e-60, 1e-7})
    EXPECT_THROW(ahs::makeRydbergModel(makeProgram({{0., 0.}, {spacing, 0.}}),
                                       ahs::fresnelCan),
                 std::invalid_argument);
  EXPECT_NO_THROW(ahs::makeRydbergModel(pair, ahs::DeviceSpecification{0.0}));
}

TEST(AHSModelTest, SamplingSiteOrderShotsAndSeed) {
  std::vector<std::complex<double>> state(8, 0.0);
  state[1] = 1.0;
  auto counts = analog::sampleStateVector(state, 37, 13);
  EXPECT_EQ(counts.count("100"), 37);
  EXPECT_EQ(counts.get_total_shots(), 37);
  auto vacancies = analog::sampleStateVector(state, 37, 13, {0, 1, 0, 1, 1});
  EXPECT_EQ(vacancies.count("01000"), 37);
  state[1] = std::sqrt(0.3);
  state[6] = std::sqrt(0.7);
  auto first = analog::sampleStateVector(state, 1000, 13);
  auto second = analog::sampleStateVector(state, 1000, 13);
  EXPECT_EQ(first.count("100"), second.count("100"));
  EXPECT_EQ(first.count("011"), second.count("011"));
  EXPECT_EQ(first.count("100") + first.count("011"), 1000);
  EXPECT_NEAR(first.probability("100"), 0.3, 0.06);
  EXPECT_EQ(analog::sampleStateVector(state, 0, 13).get_total_shots(), 0);
}

TEST(AHSModelTest, Vacancies) {
  auto program = makeProgram({{0., 0.}, {1e-6, 0.}, {6e-6, 0.}});
  program.setup.ahs_register.filling[1] = 0;
  auto model = ahs::makeRydbergModel(program, ahs::fresnelCan);
  EXPECT_EQ(model.dimensions.size(), 2);
  EXPECT_NEAR(matrixAt(model)(3, 3).real(),
              ahs::fresnelCan.rydbergC6 / std::pow(6e-6, 6), 1e-7);
  program.setup.ahs_register.filling = {0, 0, 0};
  EXPECT_TRUE(
      ahs::makeRydbergModel(program, ahs::fresnelCan).dimensions.empty());
  EXPECT_EQ(analog::sampleStateVector({1.}, 19, 7, {0, 0, 0}).count("000"), 19);
}

TEST(AHSModelTest, SpatialDrivingFieldsAdd) {
  auto program = makeProgram({{0., 0.}, {6e-6, 0.}}, 2e6);
  program.hamiltonian.drivingFields[0].amplitude.pattern =
      ahs::FieldPattern(std::vector<double>{1., 0.});
  auto second = program.hamiltonian.drivingFields[0];
  second.amplitude.pattern = ahs::FieldPattern(std::vector<double>{0., 0.5});
  second.phase.time_series.values = {std::numbers::pi / 2,
                                     std::numbers::pi / 2};
  program.hamiltonian.drivingFields.push_back(second);
  auto matrix = matrixAt(ahs::makeRydbergModel(program, ahs::fresnelCan));
  EXPECT_NEAR(std::abs(matrix(1, 0) - 1e6), 0., 1e-8);
  EXPECT_NEAR(std::abs(matrix(2, 0) - std::complex<double>(0., 5e5)), 0., 1e-8);
}

TEST(AHSModelTest, RejectInvalidPrograms) {
  auto program = makeProgram({{0., 0.}, {6e-6, 0.}});
  auto &field = program.hamiltonian.drivingFields[0].amplitude;
  field.time_series.times = {0.};
  EXPECT_THROW(ahs::makeRydbergModel(program, ahs::fresnelCan),
               std::invalid_argument);
  field.time_series.times = {0., 0.};
  EXPECT_THROW(ahs::makeRydbergModel(program, ahs::fresnelCan),
               std::invalid_argument);
  field.time_series.times = {0., 1e-6};
  field.pattern = ahs::FieldPattern(std::vector<double>{1.});
  EXPECT_THROW(ahs::makeRydbergModel(program, ahs::fresnelCan),
               std::invalid_argument);
  field.pattern = ahs::FieldPattern("uniform");
  program.setup.ahs_register.sites[1] = {0., 0.};
  EXPECT_THROW(ahs::makeRydbergModel(program, ahs::fresnelCan),
               std::invalid_argument);
  auto local = makeProgram({{0., 0.}});
  local.hamiltonian.localDetuning.push_back(
      {{ahs::TimeSeries({{0., 0.}, {4e6, 1e-6}}),
        ahs::FieldPattern(std::vector<double>{1.})}});
  EXPECT_THROW(ahs::makeRydbergModel(local, ahs::fresnelCan),
               std::invalid_argument);
}

TEST(RydbergHamiltonianTest, ConstructorValidInputs) {
  // Valid atom sites
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Valid atom filling
  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  EXPECT_EQ(hamiltonian.get_atom_sites().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_atom_filling().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_amplitude().evaluate({}),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(hamiltonian.get_phase().evaluate({}),
            std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hamiltonian.get_delta_global().evaluate({}),
            std::complex<double>(-0.5, 0.0));
}

TEST(RydbergHamiltonianTest, ConstructorWithAtomFilling) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Valid atom filling
  std::vector<int> atom_filling = {1, 0, 1};

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global,
                                  atom_filling);

  EXPECT_EQ(hamiltonian.get_atom_sites().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_atom_filling(), atom_filling);
}

TEST(RydbergHamiltonianTest, InvalidAtomFillingSize) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Invalid atom filling size
  std::vector<int> atom_filling = {1, 0};

  EXPECT_ANY_THROW(rydberg_hamiltonian(atom_sites, amplitude, phase,
                                       delta_global, atom_filling));
}

TEST(RydbergHamiltonianTest, UnsupportedLocalDetuning) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Invalid delta_local
  auto delta_local =
      std::make_pair(scalar_operator(0.5), std::vector<double>{0.1, 0.2, 0.3});

  EXPECT_ANY_THROW(rydberg_hamiltonian(atom_sites, amplitude, phase,
                                       delta_global, {}, delta_local));
}

TEST(RydbergHamiltonianTest, Accessors) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  EXPECT_EQ(hamiltonian.get_atom_sites(), atom_sites);
  EXPECT_EQ(hamiltonian.get_amplitude().evaluate({}),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(hamiltonian.get_phase().evaluate({}),
            std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hamiltonian.get_delta_global().evaluate({}),
            std::complex<double>(-0.5, 0.0));
}

TEST(RydbergHamiltonianTest, DefaultAtomFilling) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  std::vector<int> expected_filling(atom_sites.size(), 1);
  EXPECT_EQ(hamiltonian.get_atom_filling(), expected_filling);
}
