/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "NoiseModel.h"
#include "Logger.h"
#include "common/EigenDense.h"

namespace cudaq {

kraus_op &kraus_op::operator=(const kraus_op &other) {
  data = other.data;
  return *this;
}

kraus_op kraus_op::adjoint() {
  Eigen::Map<Eigen::MatrixXcd> map(data.data(), nRows, nCols);
  Eigen::MatrixXcd adj = map.adjoint();
  std::vector<complex> vec(adj.data(), adj.data() + adj.rows() * adj.cols());
  return kraus_op(vec);
}

bool isIdentity(const Eigen::MatrixXcd &mat, double threshold = 1e-9) {
  Eigen::MatrixXcd idMat = Eigen::MatrixXcd::Identity(mat.rows(), mat.cols());
  return mat.isApprox(Eigen::MatrixXcd::Identity(mat.rows(), mat.cols()),
                      threshold);
}

bool validateCPTP(const std::vector<Eigen::MatrixXcd> &mats,
                  double threshold = 1e-9) {
  if (mats.empty()) {
    return true;
  }

  Eigen::MatrixXcd cptp =
      Eigen::MatrixXcd::Zero(mats[0].rows(), mats[0].cols());
  for (const auto &mat : mats) {
    cptp = cptp + mat.adjoint() * mat;
  }
  return isIdentity(cptp, threshold);
}

void kraus_channel::validateCompleteness() {
  // First check that all the kraus_ops have the same size.
  auto size = ops[0].nRows;
  for (std::size_t i = 1; i < ops.size(); ++i)
    if (ops[i].nRows != size)
      throw std::runtime_error(
          "Kraus ops passed to this channel do not all have the same size.");

  std::vector<Eigen::MatrixXcd> matrices;
  for (auto &op : ops) {
    Eigen::Map<Eigen::MatrixXcd> map(op.data.data(), op.nRows, op.nCols);
    matrices.push_back(map);
  }
  if (!validateCPTP(matrices))
    throw std::runtime_error(
        "Provided kraus_ops are not completely positive and trace preserving.");
}

kraus_channel::kraus_channel( // std::vector<std::size_t> &qbits,
    std::vector<kraus_op> &_ops)
    : ops(_ops) {
  validateCompleteness();
}

kraus_channel::kraus_channel(const kraus_channel &other) : ops(other.ops) {}

std::size_t kraus_channel::size() const { return ops.size(); }

bool kraus_channel::empty() const { return ops.empty(); }

std::size_t kraus_channel::dimension() const { return ops[0].nRows; }

kraus_op &kraus_channel::operator[](const std::size_t idx) { return ops[idx]; }

kraus_channel &kraus_channel::operator=(const kraus_channel &other) {
  // qubits = other.qubits;
  ops = other.ops;
  return *this;
}

std::vector<kraus_op> kraus_channel::get_ops() { return ops; }
void kraus_channel::push_back(kraus_op op) { ops.push_back(op); }

void noise_model::add_channel(const std::string &quantumOp,
                              const std::vector<std::size_t> &qubits,
                              const kraus_channel &channel) {

  if (std::find(availableOps.begin(), availableOps.end(), quantumOp) ==
      availableOps.end())
    throw std::runtime_error(
        "Invalid quantum op for noise_model::add_channel (" + quantumOp + ").");

  // Check that we've been given the correct number of qubits
  auto nQubits = qubits.size();
  auto dim = 1UL << nQubits;
  auto channelDim = channel.dimension();
  if (dim != channelDim)
    throw std::runtime_error(
        "Dimension mismatch - kraus_channel with dimension = " +
        std::to_string(channelDim) + " on " + std::to_string(nQubits) +
        " qubits.");

  auto key = std::make_pair(quantumOp, qubits);
  auto iter = noiseModel.find(key);
  if (iter == noiseModel.end()) {
    cudaq::info("Adding new kraus_channel to noise_model ({}, {})", quantumOp,
                qubits);
    noiseModel.insert({key, {channel}});
    return;
  }

  cudaq::info("kraus_channel existed for {}, adding new kraus_channel to "
              "noise_model (qubits = {})",
              quantumOp, qubits);

  iter->second.push_back(channel);
}

std::vector<kraus_channel>
noise_model::get_channels(const std::string &quantumOp,
                          const std::vector<std::size_t> &qubits) const {
  auto key = std::make_pair(quantumOp, qubits);
  auto iter = noiseModel.find(key);
  if (iter == noiseModel.end()) {
    cudaq::info("No kraus_channel available for {} on {}.", quantumOp, qubits);
    return {};
  }

  cudaq::info("Found kraus_channel for {} on {}.", quantumOp, qubits);
  return iter->second;
}

depolarization_channel::depolarization_channel(const double p)
    : kraus_channel() {
  std::vector<complex> k0v{std::sqrt(1 - p), 0, 0, std::sqrt(1 - p)},
      k1v{0, std::sqrt(p / 3.), std::sqrt(p / 3.), 0},
      k2v{0, cudaq::complex{0, -1. * std::sqrt(p / 3.)},
          cudaq::complex{0, std::sqrt(p / 3.)}, 0},
      k3v{std::sqrt(p / 3.), 0, 0, -1. * std::sqrt(p / 3.)};
  ops = {k0v, k1v, k2v, k3v};
  validateCompleteness();
}

amplitude_damping_channel::amplitude_damping_channel(const double p)
    : kraus_channel() {
  std::vector<complex> k0v{1, 0, 0, std::sqrt(1 - p)},
      k1v{0, 0, std::sqrt(p), 0};
  ops = {k0v, k1v};
  validateCompleteness();
}

bit_flip_channel::bit_flip_channel(const double p) : kraus_channel() {
  std::vector<complex> k0v{std::sqrt(1 - p), 0, 0, std::sqrt(1 - p)},
      k1v{0, std::sqrt(p), std::sqrt(p), 0};
  ops = {k0v, k1v};
  validateCompleteness();
}

phase_flip_channel::phase_flip_channel(const double p) : kraus_channel() {
  std::vector<complex> k0v{std::sqrt(1 - p), 0, 0, std::sqrt(1 - p)},
      k1v{std::sqrt(p), 0, 0, -1. * std::sqrt(p)};
  ops = {k0v, k1v};
  validateCompleteness();
}
} // namespace cudaq
