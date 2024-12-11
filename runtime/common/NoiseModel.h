/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"

#include <array>
#include <complex>
#include <functional>
#include <math.h>
#include <unordered_map>
#include <vector>

namespace cudaq {

/// @brief Noise model enumerated type that allows downstream simulators of
/// `kraus_channel` objects to apply simulator-specific logic for well-known
/// noise models.
enum class noise_model_type {
  unknown,
  depolarization_channel,
  amplitude_damping_channel,
  bit_flip_channel,
  phase_flip_channel
};

/// @brief A kraus_op represents a single Kraus operation,
/// described as a complex matrix of specific size. The matrix
/// is represented here as a 1d array (specifically a std::vector).
struct kraus_op {

  /// @brief Matrix data, represented as a 1d flattened
  // *row major* matrix.
  std::vector<cudaq::complex> data;

  /// @brief The number of rows in the matrix
  std::size_t nRows = 0;

  /// @brief The number of columns in the matrix
  /// NOTE we currently assume nRows == nCols
  std::size_t nCols = 0;

  /// @brief Copy constructor
  kraus_op(const kraus_op &) = default;

  /// @brief Constructor, initialize from vector data
  kraus_op(std::vector<cudaq::complex> d) : data(d) {
    auto nElements = d.size();
    auto sqrtNEl = std::sqrt(nElements);
    if (sqrtNEl * sqrtNEl != nElements)
      throw std::runtime_error(
          "Invalid number of elements to kraus_op. Must be square.");

    nRows = (std::size_t)std::round(sqrtNEl);
    nCols = nRows;
  }

  /// @brief Constructor, initialize from initializer_list
  template <typename T>
  kraus_op(std::initializer_list<T> &&initList)
      : data(initList.begin(), initList.end()) {
    auto nElements = initList.size();
    auto sqrtNEl = std::sqrt(nElements);
    if (sqrtNEl * sqrtNEl != nElements)
      throw std::runtime_error(
          "Invalid number of elements to kraus_op. Must be square.");

    nRows = (std::size_t)std::round(sqrtNEl);
    nCols = nRows;
  }

  /// @brief Set this kraus_op equal to the other
  kraus_op &operator=(const kraus_op &other) {
    data = other.data;
    return *this;
  }

  /// @brief Return the adjoint of this kraus_op
  kraus_op adjoint() const {
    std::size_t N = data.size();
    std::vector<cudaq::complex> newData(N);
    for (std::size_t i = 0; i < nRows; i++)
      for (std::size_t j = 0; j < nCols; j++)
        newData[i * nRows + j] = std::conj(data[j * nCols + i]);
    return kraus_op(newData);
  }
};

void validateCompletenessRelation_fp32(const std::vector<kraus_op> &ops);
void validateCompletenessRelation_fp64(const std::vector<kraus_op> &ops);

/// @brief A kraus_channel represents a quantum noise channel
/// on specific qubits. The action of the noise channel is
/// described by a vector of Kraus operations - matrices with
/// size equal to 2**nQubits x 2**nQubits, where the number of
/// qubits is the number of provided qubit indices at construction.
class kraus_channel {
protected:
  /// @brief The qubits this kraus_channel operates on
  // std::vector<std::size_t> qubits;

  /// @brief The kraus_ops that make up this channel
  std::vector<kraus_op> ops;

  /// @brief Validate that Sum K_i^† K_i = I
  // Important: as this function dispatches different implementations based on
  // `cudaq::complex`, which is a pre-processor define, do not call this in
  // `NoiseModel.cpp`. `NoiseModel.cpp` is compiled as a `cudaq-common` library,
  // which is not aware of the backend complex type.
  void validateCompleteness() {
    if constexpr (std::is_same_v<cudaq::complex::value_type, float>) {
      validateCompletenessRelation_fp32(ops);
      return;
    }
    validateCompletenessRelation_fp64(ops);
  }

public:
  /// @brief Noise type enumeration
  noise_model_type noise_type = noise_model_type::unknown;

  /// @brief Noise parameter values
  // Use `double` as the uniform type to store channel parameters (for both
  // single- and double-precision channel definitions). Some
  // `kraus_channel` methods, e.g., copy constructor/assignment, are implemented
  // in `NoiseModel.cpp`, which is compiled to `cudaq-common` with
  // double-precision configuration regardless of the backends.
  // Hence, having a templated `parameters` member variable may cause data
  // corruption.
  std::vector<double> parameters;

  ~kraus_channel() = default;

  /// @brief The nullary constructor
  kraus_channel() = default;

  /// @brief The copy constructor
  /// @param other
  kraus_channel(const kraus_channel &other);

  /// @brief The constructor, initializes kraus_ops internally
  /// from the provided initializer_lists.
  template <typename... T>
  kraus_channel(std::initializer_list<T> &&...inputLists) {
    (ops.emplace_back(std::move(inputLists)), ...);
    validateCompleteness();
  }

  /// @brief The constructor, take qubits and channel kraus_ops as lvalue
  /// reference
  kraus_channel(const std::vector<kraus_op> &inOps) : ops(inOps) {
    validateCompleteness();
  }

  /// @brief The constructor, take qubits and channel kraus_ops as rvalue
  /// reference
  kraus_channel(std::vector<kraus_op> &&ops) : kraus_channel(ops) {}

  /// @brief Return the number of kraus_ops that make up this channel
  std::size_t size() const;

  std::size_t dimension() const;

  /// @brief Return true if there are no ops in this channel
  bool empty() const;

  /// @brief Return the kraus_op at the given index
  kraus_op &operator[](const std::size_t idx);

  /// @brief Set this kraus_channel equal to the given one.
  kraus_channel &operator=(const kraus_channel &other);

  /// @brief Return all kraus_ops in this channel
  std::vector<kraus_op> get_ops();

  /// @brief Add a kraus_op to this channel.
  void push_back(kraus_op op);
};

/// @brief The noise_model type keeps track of a set of
/// kraus_channels to be applied after the execution of
/// quantum operations. Each quantum operation maps
/// to a Kraus channel containing a number of kraus_ops to
/// be applied to the density matrix representation of the state.
class noise_model {
public:
  /// @brief Callback function type for noise channel.
  /// Given the qubit operands and gate parameters, this function should return
  /// a concrete noise channel.
  using PredicateFuncTy = std::function<kraus_channel(
      const std::vector<std::size_t> &, const std::vector<double> &)>;

protected:
  /// @brief Noise Model data map key is a (quantum Op + qubits applied to)
  using KeyT = std::pair<std::string, std::vector<std::size_t>>;

  /// @brief unordered_map will need a custom hash function
  struct KeyTHash {
    template <class T, class U>
    std::size_t operator()(const std::pair<T, U> &p) const {
      auto hash = std::hash<T>{}(p.first);
      hash ^= p.second.size();
      for (auto &i : p.second) {
        hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  /// @brief Useful typedef for the noise model data map
  using NoiseModelOpMap =
      std::unordered_map<KeyT, std::vector<kraus_channel>, KeyTHash>;

  // The noise model is a mapping of quantum operation
  // names to a Kraus channel applied after the operation is applied.
  NoiseModelOpMap noiseModel;

  /// @brief Gate identity for a match-all condition.
  // In this case, it will match an operation with any qubits.
  // The controlled versions of a gate are tracked by the number of control
  // qubits.
  struct GateIdentifier {
    std::string name;
    std::size_t numControls;
    bool operator==(const GateIdentifier &other) const {
      return other.name == name && other.numControls == numControls;
    };
  };

  // In addition to specific (gate + operands) map, we have a default map,
  // which tracks noise channels attached to all operations of that type.
  // This map is keyed by the gate-name + number of control bits, e.g., x(1)
  // means cnot.
  struct GateIdentifierHash {
    std::size_t operator()(const GateIdentifier &p) const {
      const std::string fullName =
          p.name + "(" + std::to_string(p.numControls) + ")";
      return std::hash<std::string>{}(fullName);
    }
  };

  /// @brief Useful typedef for the noise model data map
  using DefaultNoiseModelOpMap =
      std::unordered_map<GateIdentifier, std::vector<kraus_channel>,
                         GateIdentifierHash>;
  /// @brief  Matched-all noise channel map
  DefaultNoiseModelOpMap defaultNoiseModel;

  /// @brief Noise model by callback function map
  std::unordered_map<std::string, PredicateFuncTy> gatePredicates;

  static constexpr const char *availableOps[] = {
      "x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "r1", "u3", "mz"};

public:
  /// @brief default constructor
  noise_model() = default;

  /// @brief Return true if there are no kraus_channels in this noise model.
  /// @return
  bool empty() const {
    return noiseModel.empty() && defaultNoiseModel.empty() &&
           gatePredicates.empty();
  }

  /// @brief Add the Kraus channel to the specified one-qubit quantum
  /// operation. It applies to the quantumOp operation for the specified
  /// qubits in the kraus_channel.
  void add_channel(const std::string &quantumOp,
                   const std::vector<std::size_t> &qubits,
                   const kraus_channel &channel);

  /// @brief Add the Kraus channel as a callback to the specified quantum
  /// operation.
  // The callback function will be called with the gate operands and gate
  // parameters whenever the specified quantum operation is executed. The
  // callback function should return a concrete noise channel. This can be an
  // empty noise channel if no noise is expected.
  /// @param quantumOp Quantum operation that the noise channel applies to.
  /// @param pred Callback function that generates a noise channel.
  void add_channel(const std::string &quantumOp, const PredicateFuncTy &pred);

  /// @brief Add the Kraus channel that applies to a quantum operation on any
  /// arbitrary qubits.
  /// @param quantumOp Quantum operation that the noise channel applies to.
  /// @param channel The Kraus channel to apply.
  /// @param numControls Number of control qubits for the gate. Default is 0
  /// (gate without a control modifier).
  void add_all_qubit_channel(const std::string &quantumOp,
                             const kraus_channel &channel, int numControls = 0);

  /// @brief Add the provided kraus_channel to all
  /// specified quantum operations.
  template <typename... QuantumOp>
  void add_channel(const std::vector<std::size_t> &qubits,
                   const kraus_channel &channel) {
    std::vector<std::string> names;
    std::apply(
        [&](const auto &...elements) { (names.push_back(elements.name), ...); },
        std::tuple<QuantumOp...>());
    for (auto &name : names)
      add_channel(name, qubits, channel);
  }

  /// @brief Add the provided kraus_channel callback to all
  /// specified quantum operations.
  template <typename... QuantumOp>
  void add_channel(const PredicateFuncTy &pred) {
    std::vector<std::string> names;
    std::apply(
        [&](const auto &...elements) { (names.push_back(elements.name), ...); },
        std::tuple<QuantumOp...>());
    for (auto &name : names)
      add_channel(name, pred);
  }

  /// @brief Add the provided kraus_channel to all
  /// specified quantum operations applying on arbitrary qubits.
  template <typename... QuantumOp>
  void add_all_qubit_channel(const kraus_channel &channel,
                             int numControls = 0) {
    std::vector<std::string> names;
    std::apply(
        [&](const auto &...elements) { (names.push_back(elements.name), ...); },
        std::tuple<QuantumOp...>());
    for (auto &name : names)
      add_all_qubit_channel(name, channel, numControls);
  }

  /// @brief Return relevant kraus_channels on the specified qubits for
  // the given quantum operation. This will merge Kraus channels
  // that exists for the same quantumOp and qubits.
  std::vector<kraus_channel>
  get_channels(const std::string &quantumOp,
               const std::vector<std::size_t> &targetQubits,
               const std::vector<std::size_t> &controlQubits = {},
               const std::vector<double> &params = {}) const;

  /// @brief Get all kraus_channels on the given qubits
  template <typename QuantumOp>
  std::vector<kraus_channel>
  get_channels(const std::vector<std::size_t> &targetQubits,
               const std::vector<std::size_t> &controlQubits = {},
               const std::vector<double> &params = {}) const {
    QuantumOp op;
    return get_channels(op.name, targetQubits, controlQubits, params);
  }
};

/// @brief depolarization_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit depolarization error channel.
class depolarization_channel : public kraus_channel {
public:
  depolarization_channel(const real probability) : kraus_channel() {
    auto three = static_cast<real>(3.);
    auto negOne = static_cast<real>(-1.);
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{0, std::sqrt(probability / three), std::sqrt(probability / three),
            0},
        k2v{0, cudaq::complex{0, negOne * std::sqrt(probability / three)},
            cudaq::complex{0, std::sqrt(probability / three)}, 0},
        k3v{std::sqrt(probability / three), 0, 0,
            negOne * std::sqrt(probability / three)};
    ops = {k0v, k1v, k2v, k3v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::depolarization_channel;
    validateCompleteness();
  }
};

/// @brief amplitude_damping_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit amplitude damping error channel.
class amplitude_damping_channel : public kraus_channel {
public:
  amplitude_damping_channel(const real probability) : kraus_channel() {
    std::vector<cudaq::complex> k0v{1, 0, 0, std::sqrt(1 - probability)},
        k1v{0, std::sqrt(probability), 0, 0};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::amplitude_damping_channel;
    validateCompleteness();
  }
};

/// @brief bit_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit bit flipping error channel.
class bit_flip_channel : public kraus_channel {
public:
  bit_flip_channel(const real probability) : kraus_channel() {
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{0, std::sqrt(probability), std::sqrt(probability), 0};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::bit_flip_channel;
    validateCompleteness();
  }
};

/// @brief phase_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit phase flip error channel.
class phase_flip_channel : public kraus_channel {
public:
  phase_flip_channel(const real probability) : kraus_channel() {
    auto negOne = static_cast<real>(-1.);
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{std::sqrt(probability), 0, 0, negOne * std::sqrt(probability)};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::phase_flip_channel;
    validateCompleteness();
  }
};
} // namespace cudaq
