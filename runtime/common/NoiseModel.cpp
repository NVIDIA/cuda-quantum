/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "NoiseModel.h"
#include "FmtCore.h"
#include "Logger.h"
#include "common/CustomOp.h"
#include "common/EigenDense.h"
#include <numeric>
#include <optional>

namespace cudaq {

// Helper to check whether a matrix is a scaled unitary matrix, i.e., `k * U`
// where U is a unitary matrix. If so, it also returns the `k` factor.
// Otherwise, return a nullopt.
static std::optional<double>
isScaledUnitary(const std::vector<std::complex<double>> &mat, double eps) {
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatTy;
  const int dim = std::log2(mat.size());
  Eigen::Map<const RowMajorMatTy> kMat(mat.data(), dim, dim);
  if (kMat.isZero(eps))
    return 0.0;
  // Check that (K_dag * K) is a scaled identity matrix
  // i.e., the K matrix is a scaled unitary.
  auto kdK = kMat.adjoint() * kMat;
  if (!kdK.isDiagonal(eps))
    return std::nullopt;
  // First element
  std::complex<double> val = kdK(0, 0);
  if (val.real() > 0.0 && std::abs(val.imag()) < eps) {
    auto scaledKdK = (std::complex<double>{1.0} / val) * kdK;
    if (scaledKdK.isIdentity(eps))
      return std::sqrt(val.real());
  }
  return std::nullopt;
}

// Helper to determine if a vector of Kraus ops are actually a unitary mixture.
// If so, it returns all the unitaries and the probabilities associated with
// each one of those unitaries.
static std::optional<std::pair<std::vector<double>,
                               std::vector<std::vector<std::complex<double>>>>>
computeUnitaryMixture(
    const std::vector<std::vector<std::complex<double>>> &krausOps,
    double tol = 1e-6) {
  std::vector<double> probs;
  std::vector<std::vector<std::complex<double>>> mats;
  const auto scaleMat = [](const std::vector<std::complex<double>> &mat,
                           double scaleFactor) {
    std::vector<std::complex<double>> scaledMat = mat;
    // If scaleFactor is 0, then it means the original matrix was likely all
    // zeros. In that case, the probability will be 0, so the matrix doesn't
    // matter, but we don't want NaNs to trickle in anywhere.
    if (scaleFactor != 0.0)
      for (auto &x : scaledMat)
        x /= scaleFactor;
    return scaledMat;
  };
  for (const auto &op : krausOps) {
    const auto scaledFactor = isScaledUnitary(op, tol);
    if (!scaledFactor.has_value())
      return std::nullopt;
    probs.emplace_back(scaledFactor.value() * scaledFactor.value());
    mats.emplace_back(scaleMat(op, scaledFactor.value()));
  }

  if (std::abs(1.0 - std::reduce(probs.begin(), probs.end())) > tol)
    return std::nullopt;

  return std::make_pair(probs, mats);
}

template <typename EigenMatTy>
bool isIdentity(const EigenMatTy &mat, double threshold = 1e-9) {
  EigenMatTy idMat = EigenMatTy::Identity(mat.rows(), mat.cols());
  return mat.isApprox(EigenMatTy::Identity(mat.rows(), mat.cols()), threshold);
}

template <typename EigenMatTy>
bool validateCPTP(const std::vector<EigenMatTy> &mats,
                  double threshold = 1e-4) {
  if (mats.empty()) {
    return true;
  }

  EigenMatTy cptp = EigenMatTy::Zero(mats[0].rows(), mats[0].cols());
  for (const auto &mat : mats) {
    cptp = cptp + mat.adjoint() * mat;
  }
  return isIdentity(cptp, threshold);
}

void validateCompletenessRelation_fp32(const std::vector<kraus_op> &ops) {
  // First check that all the kraus_ops have the same size.
  auto size = ops[0].nRows;
  for (std::size_t i = 1; i < ops.size(); ++i)
    if (ops[i].nRows != size)
      throw std::runtime_error(
          "Kraus ops passed to this channel do not all have the same size.");
  typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatTy;
  std::vector<RowMajorMatTy> matrices;
  for (auto &op : ops) {
    auto *nonConstPtr = const_cast<complex *>(op.data.data());
    Eigen::Map<RowMajorMatTy> map(
        reinterpret_cast<std::complex<float> *>(nonConstPtr), op.nRows,
        op.nCols);
    matrices.push_back(map);
  }
  if (!validateCPTP(matrices, 1e-4))
    throw std::runtime_error(
        "Provided kraus_ops are not completely positive and trace preserving.");
}

void validateCompletenessRelation_fp64(const std::vector<kraus_op> &ops) {
  // First check that all the kraus_ops have the same size.
  auto size = ops[0].nRows;
  for (std::size_t i = 1; i < ops.size(); ++i)
    if (ops[i].nRows != size)
      throw std::runtime_error(
          "Kraus ops passed to this channel do not all have the same size.");
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatTy;
  std::vector<RowMajorMatTy> matrices;
  for (auto &op : ops) {
    auto *nonConstPtr = const_cast<complex *>(op.data.data());
    Eigen::Map<RowMajorMatTy> map(
        reinterpret_cast<std::complex<double> *>(nonConstPtr), op.nRows,
        op.nCols);
    matrices.push_back(map);
  }
  if (!validateCPTP(matrices))
    throw std::runtime_error(
        "Provided kraus_ops are not completely positive and trace preserving.");
}

void generateUnitaryParameters_fp32(
    const std::vector<kraus_op> &ops,
    std::vector<std::vector<std::complex<double>>> &unitary_ops,
    std::vector<double> &probabilities) {
  std::vector<std::vector<std::complex<double>>> double_kraus_ops;
  double_kraus_ops.reserve(ops.size());
  for (auto &op : ops) {
    // WARNING: danger here. We are intentially treating the incoming op as fp32
    // type instead of what the compiler thinks it is (fp64). We have to do this
    // because this file is compiled with cudaq::real = fp64, but the incoming
    // data for this specific routine is actually fp32.
    const std::complex<float> *ptr =
        reinterpret_cast<const std::complex<float> *>(op.data.data());
    // Use 2 * size because pointer arithmetic is on fp32 instead of fp64
    double_kraus_ops.emplace_back(
        std::vector<std::complex<double>>(ptr, ptr + 2 * op.data.size()));
  }

  auto asUnitaryMixture = computeUnitaryMixture(double_kraus_ops);
  if (asUnitaryMixture.has_value()) {
    probabilities = std::move(asUnitaryMixture.value().first);
    unitary_ops = std::move(asUnitaryMixture.value().second);
  }
}

void generateUnitaryParameters_fp64(
    const std::vector<kraus_op> &ops,
    std::vector<std::vector<std::complex<double>>> &unitary_ops,
    std::vector<double> &probabilities) {
  std::vector<std::vector<std::complex<double>>> double_kraus_ops;
  double_kraus_ops.reserve(ops.size());
  for (auto &op : ops)
    double_kraus_ops.emplace_back(
        std::vector<std::complex<double>>(op.data.begin(), op.data.end()));

  auto asUnitaryMixture = computeUnitaryMixture(double_kraus_ops);
  if (asUnitaryMixture.has_value()) {
    probabilities = std::move(asUnitaryMixture.value().first);
    unitary_ops = std::move(asUnitaryMixture.value().second);
  }
}

kraus_channel::kraus_channel(const kraus_channel &other)
    : ops(other.ops), noise_type(other.noise_type),
      parameters(other.parameters), unitary_ops(other.unitary_ops),
      probabilities(other.probabilities) {}

std::size_t kraus_channel::size() const { return ops.size(); }

bool kraus_channel::empty() const { return ops.empty(); }

std::size_t kraus_channel::dimension() const { return ops[0].nRows; }

kraus_op &kraus_channel::operator[](const std::size_t idx) { return ops[idx]; }

kraus_channel &kraus_channel::operator=(const kraus_channel &other) {
  ops = other.ops;
  noise_type = other.noise_type;
  parameters = other.parameters;
  unitary_ops = other.unitary_ops;
  probabilities = other.probabilities;
  return *this;
}

std::vector<kraus_op> kraus_channel::get_ops() const { return ops; }
void kraus_channel::push_back(kraus_op op) { ops.push_back(op); }

void noise_model::add_channel(const std::string &quantumOp,
                              const std::vector<std::size_t> &qubits,
                              const kraus_channel &channel) {

  if (std::find(std::begin(availableOps), std::end(availableOps), quantumOp) ==
          std::end(availableOps) &&
      !customOpRegistry::getInstance().isOperationRegistered(quantumOp))
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
    CUDAQ_INFO("Adding new kraus_channel to noise_model ({}, {})", quantumOp,
               qubits);
    noiseModel.insert({key, {channel}});
    return;
  }

  CUDAQ_INFO("kraus_channel existed for {}, adding new kraus_channel to "
             "noise_model (qubits = {})",
             quantumOp, qubits);

  iter->second.push_back(channel);
}

void noise_model::add_all_qubit_channel(const std::string &quantumOp,
                                        const kraus_channel &channel,
                                        int numControls) {
  auto actualGateName = quantumOp;
  const bool isCustomOp =
      customOpRegistry::getInstance().isOperationRegistered(actualGateName);
  if (numControls == 0 && quantumOp.starts_with('c') && !isCustomOp) {
    // Infer the number of control bits from gate name (with 'c' prefixes)
    // Note: We only support up to 2 control bits using this notation, e.g.,
    // 'cx', 'ccx'. Users will need to use the numControls parameter for more
    // complex cases.
    // Note: this convention doesn't apply to custom operations.
    numControls = quantumOp.starts_with("cc") ? 2 : 1;
    actualGateName = quantumOp.substr(numControls);
    if (actualGateName.starts_with('c'))
      throw std::runtime_error(
          "Controlled gates with more than 2 control bits must be specified "
          "using the numControls parameter.");
  }

  if (std::find(std::begin(availableOps), std::end(availableOps),
                actualGateName) == std::end(availableOps) &&
      !isCustomOp)
    throw std::runtime_error(
        "Invalid quantum op for noise_model::add_channel (" + quantumOp + ").");
  GateIdentifier key(actualGateName, numControls);
  auto iter = defaultNoiseModel.find(key);
  if (iter == defaultNoiseModel.end()) {
    CUDAQ_INFO("Adding new all-qubit kraus_channel to noise_model ({}, number "
               "of control bits = {})",
               actualGateName, numControls);
    defaultNoiseModel.insert({key, {channel}});
    return;
  }

  CUDAQ_INFO("kraus_channel existed for {}, adding new kraus_channel to "
             "noise_model (number of control bits = {})",
             actualGateName, numControls);

  iter->second.push_back(channel);
}

void noise_model::add_channel(const std::string &quantumOp,
                              const PredicateFuncTy &pred) {
  if (std::find(std::begin(availableOps), std::end(availableOps), quantumOp) ==
          std::end(availableOps) &&
      !customOpRegistry::getInstance().isOperationRegistered(quantumOp))
    throw std::runtime_error(
        "Invalid quantum op for noise_model::add_channel (" + quantumOp + ").");
  auto iter = gatePredicates.find(quantumOp);
  if (iter == gatePredicates.end()) {
    CUDAQ_INFO("Adding new callback kraus_channel to noise_model for {}.",
               quantumOp);
    gatePredicates.insert({quantumOp, pred});
    return;
  }

  throw std::logic_error("An callback kraus_channel has been defined for " +
                         quantumOp + " gate.");
}

std::vector<kraus_channel>
noise_model::get_channels(const std::string &quantumOp,
                          const std::vector<std::size_t> &targetQubits,
                          const std::vector<std::size_t> &controlQubits,
                          const std::vector<double> &params) const {
  std::vector<std::size_t> qubits{controlQubits.begin(), controlQubits.end()};
  qubits.insert(qubits.end(), targetQubits.begin(), targetQubits.end());
  const auto verifyChannelDimension =
      [&](const std::vector<kraus_channel> &channels) {
        auto nQubits = qubits.size();
        auto dim = 1UL << nQubits;
        return std::all_of(
            channels.begin(), channels.end(), [dim](const auto &channel) {
              return channel.empty() || channel.dimension() == dim;
            });
      };

  std::vector<kraus_channel> resultChannels;
  // Search qubit-specific noise settings
  auto key = std::make_pair(quantumOp, qubits);
  auto iter = noiseModel.find(key);
  // Note: we've validated the channel dimension in the 'add_channel' method.
  if (iter != noiseModel.end()) {
    CUDAQ_INFO("Found kraus_channel for {} on {}.", quantumOp, qubits);
    const auto &krausChannel = iter->second;
    resultChannels.insert(resultChannels.end(), krausChannel.begin(),
                          krausChannel.end());
  }

  // Look up default noise channel
  auto defaultIter =
      defaultNoiseModel.find(GateIdentifier(quantumOp, controlQubits.size()));
  if (defaultIter != defaultNoiseModel.end()) {
    CUDAQ_INFO(
        "Found default kraus_channel setting for {} with {} control bits.",
        quantumOp, controlQubits.size());

    if (!verifyChannelDimension(defaultIter->second))
      throw std::runtime_error(
          fmt::format("Dimension mismatch: all-qubit kraus_channel with for "
                      "{} with {} control qubits encountered unexpected "
                      "kraus operator dimension (expecting dimension of {}).",
                      quantumOp, controlQubits.size(), 1UL << qubits.size()));

    const auto &krausChannel = defaultIter->second;
    resultChannels.insert(resultChannels.end(), krausChannel.begin(),
                          krausChannel.end());
  }

  // Look up predicate-specific noise settings
  auto predIter = gatePredicates.find(quantumOp);
  if (predIter != gatePredicates.end()) {
    CUDAQ_INFO("Found callback kraus_channel setting for {}.", quantumOp);
    const auto krausChannel = predIter->second(qubits, params);
    if (!verifyChannelDimension({krausChannel}))
      throw std::runtime_error(fmt::format(
          "Dimension mismatch: kraus_channel with for "
          "{} on qubits {} with gate parameters {} encountered unexpected "
          "kraus operator dimension (expecting dimension of {}, got {}).",
          quantumOp, qubits, params, 1UL << qubits.size(),
          krausChannel.dimension()));
    if (!krausChannel.empty())
      resultChannels.emplace_back(krausChannel);
  }

  if (resultChannels.empty())
    CUDAQ_INFO("No kraus_channel available for {} on {}.", quantumOp, qubits);

  return resultChannels;
}

noise_model::noise_model() {
  register_channel<depolarization_channel>();
  register_channel<amplitude_damping_channel>();
  register_channel<bit_flip_channel>();
  register_channel<phase_flip_channel>();
  register_channel<x_error>();
  register_channel<y_error>();
  register_channel<z_error>();
  register_channel<amplitude_damping>();
  register_channel<phase_damping>();
  register_channel<pauli1>();
  register_channel<pauli2>();
  register_channel<depolarization1>();
  register_channel<depolarization2>();
}

std::string get_noise_model_type_name(noise_model_type type) {
  return noise_model_strings[static_cast<std::size_t>(type)];
}

} // namespace cudaq
