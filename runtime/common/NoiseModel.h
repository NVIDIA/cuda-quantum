/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <complex>
#include <unordered_map>
#include <vector>

namespace cudaq {

using complex = std::complex<double>;

/// @brief A kraus_op represents a single Kraus operation,
/// described as a complex matrix of specific size. The matrix
/// is represented here as a 1d array (specifically a std::vector).
struct kraus_op {

  /// @brief Matrix data, represented as a 1d flattened
  // row major matrix.
  std::vector<std::complex<double>> data;

  /// @brief The number of rows in the matrix
  std::size_t nRows = 0;

  /// @brief The number of columns in the matrix
  /// NOTE we currently assume nRows == nCols
  std::size_t nCols = 0;

  /// @brief Copy constructor
  kraus_op(const kraus_op &) = default;

  /// @brief Constructor, initialize from vector data
  kraus_op(std::vector<complex> d) : data(d) {
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
  kraus_op &operator=(const kraus_op &other);

  /// @brief Return the adjoint of this kraus_op
  kraus_op adjoint();
};

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

  /// @brief Validate that Sum Ki^ Ki = I
  void validateCompleteness();

public:
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
  kraus_channel(std::vector<kraus_op> &ops);

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

  static const constexpr std::array<const char *, 10> availableOps{
      "x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "r1"};

  // The noise model is a mapping of quantum operation
  // names to a Kraus channel applied after the operation is applied.
  NoiseModelOpMap noiseModel;

public:
  /// @brief default constructor
  noise_model() = default;

  /// @brief Return true if there are no kraus_channels in this noise model.
  /// @return
  bool empty() const { return noiseModel.empty(); }

  /// @brief Add the Kraus channel to the specified one-qubit quantum
  /// operation. It applies to the quantumOp operation for the specified
  /// qubits in the kraus_channel.
  void add_channel(const std::string &quantumOp,
                   const std::vector<std::size_t> &qubits,
                   const kraus_channel &channel);
  void add_channel(const std::string &quantumOp,
                   const std::vector<std::size_t> &&qubits,
                   const kraus_channel &channel) {
    add_channel(quantumOp, qubits, channel);
  }

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

  /// @brief Return relevant kraus_channels on the specified qubits for
  // the given quantum operation. This will merge Kraus channels
  // that exists for the same quantumOp and qubits.
  std::vector<kraus_channel>
  get_channels(const std::string &quantumOp,
               const std::vector<std::size_t> &qubits) const;

  /// @brief Get all kraus_channels on the given qubits
  template <typename QuantumOp>
  std::vector<kraus_channel>
  get_channels(const std::vector<std::size_t> &qubits) const {
    QuantumOp op;
    return get_channels(op.name, qubits);
  }
};

/// @brief depolarization_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit depolarization error channel.
class depolarization_channel : public kraus_channel {
public:
  depolarization_channel(const double probability);
};

/// @brief amplitude_damping_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit amplitude damping error channel.
class amplitude_damping_channel : public kraus_channel {
public:
  amplitude_damping_channel(const double probability);
};

/// @brief bit_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit bit flipping error channel.
class bit_flip_channel : public kraus_channel {
public:
  bit_flip_channel(const double probability);
};

/// @brief phase_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit phase flip error channel.
class phase_flip_channel : public kraus_channel {
public:
  phase_flip_channel(const double probability);
};
} // namespace cudaq
