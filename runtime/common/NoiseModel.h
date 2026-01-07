/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <math.h>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cudaq::details {
void warn(const std::string_view msg);

/// @brief Typedef for a matrix wrapper using std::vector<cudaq::complex>
using matrix_wrapper = std::vector<cudaq::complex>;

/// @brief Compute the Kronecker product of two matrices
///
/// @param A First matrix
/// @param rowsA Number of rows in matrix A
/// @param colsA Number of columns in matrix A
/// @param B Second matrix
/// @param rowsB Number of rows in matrix B
/// @param colsB Number of columns in matrix B
/// @return matrix_wrapper Result of the Kronecker product
inline matrix_wrapper kron(const matrix_wrapper &A, int rowsA, int colsA,
                           const matrix_wrapper &B, int rowsB, int colsB) {
  matrix_wrapper C((rowsA * rowsB) * (colsA * colsB));
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsA; ++j) {
      for (int k = 0; k < rowsB; ++k) {
        for (int l = 0; l < colsB; ++l) {
          C[(i * rowsB + k) * (colsA * colsB) + (j * colsB + l)] =
              A[i * colsA + j] * B[k * colsB + l];
        }
      }
    }
  }
  return C;
}

inline matrix_wrapper scale(const cudaq::real s, const matrix_wrapper &A) {
  matrix_wrapper result;
  result.reserve(A.size());
  for (auto a : A)
    result.push_back(s * a);
  return result;
}
} // namespace cudaq::details

namespace cudaq {

// Keep the noise_model_type and noise_model_strings in sync. We don't use
// macros to work around bugs in the documentation generation.
enum class noise_model_type {
  unknown,
  depolarization_channel,
  amplitude_damping_channel,
  bit_flip_channel,
  phase_flip_channel,
  x_error,
  y_error,
  z_error,
  amplitude_damping,
  phase_damping,
  pauli1,
  pauli2,
  depolarization1,
  depolarization2
};

// Keep the noise_model_type and noise_model_strings in sync. We don't use
// macros to work around bugs in the documentation generation.
static constexpr const char *noise_model_strings[] = {
    "unknown",
    "depolarization_channel",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "phase_flip_channel",
    "x_error",
    "y_error",
    "z_error",
    "amplitude_damping",
    "phase_damping",
    "pauli1",
    "pauli2",
    "depolarization1",
    "depolarization2"};

std::string get_noise_model_type_name(noise_model_type type);

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

  /// @brief The precision of the underlying data
  // This data is populated when a `kraus_op` is created and can be used to
  // introspect `kraus_op` objects across library boundary (e.g., when dynamic
  // linking is involved).
  const cudaq::simulation_precision precision =
      std::is_same_v<cudaq::real, float> ? cudaq::simulation_precision::fp32
                                         : cudaq::simulation_precision::fp64;

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
void generateUnitaryParameters_fp32(
    const std::vector<kraus_op> &ops,
    std::vector<std::vector<std::complex<double>>> &, std::vector<double> &);
void generateUnitaryParameters_fp64(
    const std::vector<kraus_op> &ops,
    std::vector<std::vector<std::complex<double>>> &, std::vector<double> &);

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

  /// @brief If all Kraus ops are - when scaled - unitary, this holds the
  /// unitary versions of those ops. These values are always "double" regardless
  /// of whether cudaq::real is float or double.
  std::vector<std::vector<std::complex<double>>> unitary_ops;

  /// @brief If all Kraus ops are - when scaled - unitary, this holds the
  /// probabilities of those ops. These values are always "double" regardless
  /// of whether cudaq::real is float or double.
  std::vector<double> probabilities;

  virtual ~kraus_channel() = default;

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
    generateUnitaryParameters();
  }

  /// @brief The constructor, take qubits and channel kraus_ops as lvalue
  /// reference
  kraus_channel(const std::vector<kraus_op> &inOps) : ops(inOps) {
    validateCompleteness();
    generateUnitaryParameters();
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
  std::vector<kraus_op> get_ops() const;

  /// @brief Add a kraus_op to this channel.
  void push_back(kraus_op op);

  std::string get_type_name() const {
    return get_noise_model_type_name(noise_type);
  }

  /// @brief Returns whether or not this is a unitary mixture.
  bool is_unitary_mixture() const { return !unitary_ops.empty(); }

  /// @brief Checks if Kraus ops have unitary representations and saves them if
  /// they do. Users should only need to call this if they have modified the
  /// Kraus ops and want to recompute these values.
  void generateUnitaryParameters() {
    unitary_ops.clear();
    probabilities.clear();
    if constexpr (std::is_same_v<cudaq::complex::value_type, float>) {
      generateUnitaryParameters_fp32(ops, this->unitary_ops,
                                     this->probabilities);
      return;
    }
    generateUnitaryParameters_fp64(ops, this->unitary_ops, this->probabilities);
  }
};

#define REGISTER_KRAUS_CHANNEL(NAME)                                           \
  static std::intptr_t get_key() __attribute__((noinline)) {                   \
    return (std::intptr_t)std::hash<std::string>{}(std::string{NAME});         \
  }

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

  // User registered kraus channels for fine grain application
  std::unordered_map<
      std::intptr_t,
      std::variant<std::function<kraus_channel(const std::vector<float> &)>,
                   std::function<kraus_channel(const std::vector<double> &)>>>
      registeredChannels;

public:
  /// @brief default constructor
  noise_model();

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

  /// @brief SFINAE helper to enable a function only if `T` is constructible
  /// with `Args...`.
  /// @tparam T The type to check
  /// @tparam Args The argument types required for construction.
  template <typename T, typename... Args>
  using requires_constructor =
      std::enable_if_t<std::is_constructible_v<T, Args...>>;

  /// @brief Register a Kraus channel. This must be called outside of a CUDA-Q
  /// kernel before the channel can be recognized inside a CUDA-Q kernel.
  template <typename KrausChannelT,
            typename = requires_constructor<KrausChannelT,
                                            const std::vector<cudaq::real> &>>
  void register_channel() {
    registeredChannels.insert(
        {KrausChannelT::get_key(),
         [](const std::vector<cudaq::real> &params) -> kraus_channel {
           KrausChannelT userChannel(params);
           if constexpr (std::is_same_v<cudaq::real, double>)
             userChannel.parameters = params;
           else
             userChannel.parameters =
                 std::vector<double>(params.begin(), params.end());
           return userChannel;
         }});
  }

  template <typename REAL>
  void register_channel(
      std::intptr_t key,
      const std::function<kraus_channel(const std::vector<REAL> &)> &gen) {
    registeredChannels.insert({key, gen});
  }

  template <typename T, typename REAL>
  kraus_channel get_channel(const std::vector<REAL> &params) const {
    auto iter = registeredChannels.find(T::get_key());
    // per spec - caller provides noise model, but channel not registered,
    // warning generated, no channel application.
    if (iter == registeredChannels.end()) {
      details::warn("requested kraus channel not registered with this "
                    "noise_model. skipping channel application.");
      return kraus_channel();
    }

    if (std::holds_alternative<
            std::function<kraus_channel(const std::vector<REAL> &)>>(
            iter->second)) {
      return std::get<std::function<kraus_channel(const std::vector<REAL> &)>>(
          iter->second)(params);
    } else {
      // Type mismatch, e.g., the model is defined for float but params is
      // supplied as double.
      if constexpr (std::is_same_v<REAL, double>) {
        // Params was double, casted to float
        return std::get<
            std::function<kraus_channel(const std::vector<float> &)>>(
            iter->second)(std::vector<float>(params.begin(), params.end()));
      } else {
        // Params was float, casted to double
        return std::get<
            std::function<kraus_channel(const std::vector<double> &)>>(
            iter->second)(std::vector<double>(params.begin(), params.end()));
      }
    }
  }

  // de-mangled name (with namespaces) for NVQIR C API
  template <typename REAL>
  kraus_channel get_channel(const std::intptr_t &key,
                            const std::vector<REAL> &params) const {
    auto iter = registeredChannels.find(key);
    if (iter == registeredChannels.end())
      throw std::runtime_error(
          "kraus channel not registered with this noise_model.");
    if (std::holds_alternative<
            std::function<kraus_channel(const std::vector<REAL> &)>>(
            iter->second)) {
      return std::get<std::function<kraus_channel(const std::vector<REAL> &)>>(
          iter->second)(params);
    } else {
      // Type mismatch, e.g., the model is defined for float but params is
      // supplied as double.
      if constexpr (std::is_same_v<REAL, double>) {
        // Params was double, casted to float
        return std::get<
            std::function<kraus_channel(const std::vector<float> &)>>(
            iter->second)(std::vector<float>(params.begin(), params.end()));
      } else {
        // Params was float, casted to double
        return std::get<
            std::function<kraus_channel(const std::vector<double> &)>>(
            iter->second)(std::vector<double>(params.begin(), params.end()));
      }
    }
  }

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
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  depolarization_channel(const std::vector<cudaq::real> &ps) {
    auto three = static_cast<real>(3.);
    auto negOne = static_cast<real>(-1.);
    auto probability = ps[0];
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
    generateUnitaryParameters();
  }
  depolarization_channel(const real probability)
      : depolarization_channel(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::depolarization_channel])
};

/// @brief amplitude_damping_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit amplitude damping error channel.
class amplitude_damping_channel : public kraus_channel {
public:
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  amplitude_damping_channel(const std::vector<cudaq::real> &ps) {
    auto probability = ps[0];
    std::vector<cudaq::complex> k0v{1, 0, 0, std::sqrt(1 - probability)},
        k1v{0, std::sqrt(probability), 0, 0};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::amplitude_damping_channel;
    validateCompleteness();
    // Note: amplitude damping is non-unitary, so there is no value in calling
    // generateUnitaryParameters().
  }
  amplitude_damping_channel(const real probability)
      : amplitude_damping_channel(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::amplitude_damping_channel])
};

/// @brief bit_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit bit flipping error channel.
class bit_flip_channel : public kraus_channel {
public:
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  bit_flip_channel(const std::vector<cudaq::real> &p) {
    cudaq::real probability = p[0];
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{0, std::sqrt(probability), std::sqrt(probability), 0};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::bit_flip_channel;
    validateCompleteness();
    generateUnitaryParameters();
  }
  bit_flip_channel(const real probability)
      : bit_flip_channel(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::bit_flip_channel])
};

/// @brief phase_flip_channel is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit phase flip error channel.
class phase_flip_channel : public kraus_channel {
public:
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  phase_flip_channel(const std::vector<cudaq::real> &p) {
    cudaq::real probability = p[0];
    auto negOne = static_cast<real>(-1.);
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{std::sqrt(probability), 0, 0, negOne * std::sqrt(probability)};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::phase_flip_channel;
    validateCompleteness();
    generateUnitaryParameters();
  }
  phase_flip_channel(const real probability)
      : phase_flip_channel(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::phase_flip_channel])
};

/// @brief amplitude_damping is the same as amplitude_damping_channel.
class amplitude_damping : public amplitude_damping_channel {
public:
  amplitude_damping(const std::vector<cudaq::real> &p)
      : amplitude_damping_channel(p) {
    noise_type = noise_model_type::amplitude_damping;
  }
  amplitude_damping(const real probability)
      : amplitude_damping_channel(probability) {
    noise_type = noise_model_type::amplitude_damping;
  }
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::amplitude_damping])
};

/// @brief phase_damping is a kraus_channel that
/// automates the creation of the kraus_ops that make up
/// a single-qubit phase damping error channel.
class phase_damping : public kraus_channel {
public:
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  phase_damping(const std::vector<cudaq::real> &ps) {
    auto probability = ps[0];
    std::vector<cudaq::complex> k0v{1, 0, 0, std::sqrt(1 - probability)},
        k1v{0, 0, 0, std::sqrt(probability)};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::phase_damping;
    validateCompleteness();
    // Note: phase damping is non-unitary, so there is no value in calling
    // generateUnitaryParameters().
  }
  phase_damping(const real probability)
      : phase_damping(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::phase_damping])
};

/// @brief z_error is a Pauli error that applies the Z operator when an error
/// occurs. It is the same as phase_flip_channel.
class z_error : public phase_flip_channel {
public:
  z_error(const std::vector<cudaq::real> &p) : phase_flip_channel(p) {
    noise_type = noise_model_type::z_error;
  }
  z_error(const real probability) : phase_flip_channel(probability) {
    noise_type = noise_model_type::z_error;
  }
  REGISTER_KRAUS_CHANNEL(noise_model_strings[(int)noise_model_type::z_error])
};

/// @brief x_error is a Pauli error that applies the X operator when an error
/// occurs. It is the same as bit_flip_channel.
class x_error : public bit_flip_channel {
public:
  x_error(const std::vector<cudaq::real> &p) : bit_flip_channel(p) {
    noise_type = noise_model_type::x_error;
  }
  x_error(const real probability) : bit_flip_channel(probability) {
    noise_type = noise_model_type::x_error;
  }
  REGISTER_KRAUS_CHANNEL(noise_model_strings[(int)noise_model_type::x_error])
};

/// @brief Y_error is a Pauli error that applies the Y operator when an error
/// occurs.
class y_error : public kraus_channel {
public:
  /// @brief Number of parameters. The one parameter is the probability that the
  /// error will occur.
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;
  y_error(const std::vector<cudaq::real> &p) {
    cudaq::real probability = p[0];
    std::complex<cudaq::real> i{0, 1};
    std::vector<cudaq::complex> k0v{std::sqrt(1 - probability), 0, 0,
                                    std::sqrt(1 - probability)},
        k1v{0, -i * std::sqrt(probability), i * std::sqrt(probability), 0};
    ops = {k0v, k1v};
    this->parameters.push_back(probability);
    noise_type = noise_model_type::y_error;
    validateCompleteness();
    generateUnitaryParameters();
  }
  y_error(const real probability)
      : y_error(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(noise_model_strings[(int)noise_model_type::y_error])
};

/// @brief A single-qubit Pauli error that applies either an X error, Y error,
/// or Z error, with 3 probabilities specified as inputs
class pauli1 : public kraus_channel {
public:
  /// @brief Number of parameters. The 3 parameters are the probability that an
  /// X error, Y error, or Z error happens. Only 1 of the 3 possible errors will
  /// happen (at most).
  constexpr static std::size_t num_parameters = 3;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 1;

  /// @brief Construct a single-qubit Pauli error Kraus channel
  /// @param p Error probabilities for X, Y, and Z errors, respectively
  pauli1(const std::vector<cudaq::real> &p) {
    if (p.size() != num_parameters)
      throw std::runtime_error(
          "Invalid number of elements to pauli1 constructor. Must be 3.");
    cudaq::real sum = 0;
    for (auto pp : p) {
      if (pp < 0)
        throw std::runtime_error("Probabilities cannot be negative");
      sum += pp;
    }
    // This is just a first-level error check. Additional checks are done in
    // validateCompleteness.
    if (sum > static_cast<cudaq::real>(1.0 + 1e-6))
      throw std::runtime_error("Sum of pauli1 parameters is >1. Must be <= 1.");

    std::complex<cudaq::real> i{0, 1};
    cudaq::details::matrix_wrapper I({1, 0, 0, 1});
    cudaq::details::matrix_wrapper X({0, 1, 1, 0});
    cudaq::details::matrix_wrapper Y({0, -i, i, 0});
    cudaq::details::matrix_wrapper Z({1, 0, 0, -1});
    cudaq::real p0 =
        std::sqrt(std::max(static_cast<cudaq::real>(1.0 - p[0] - p[1] - p[2]),
                           static_cast<cudaq::real>(0)));
    cudaq::real px = std::sqrt(p[0]);
    cudaq::real py = std::sqrt(p[1]);
    cudaq::real pz = std::sqrt(p[2]);
    std::vector<cudaq::complex> k0v = details::scale(p0, I);
    std::vector<cudaq::complex> k1v = details::scale(px, X);
    std::vector<cudaq::complex> k2v = details::scale(py, Y);
    std::vector<cudaq::complex> k3v = details::scale(pz, Z);
    ops = {k0v, k1v, k2v, k3v};
    this->parameters.reserve(p.size());
    for (auto pp : p)
      this->parameters.push_back(pp);
    noise_type = cudaq::noise_model_type::pauli1;
    validateCompleteness();
    generateUnitaryParameters();
  }
  REGISTER_KRAUS_CHANNEL(noise_model_strings[(int)noise_model_type::pauli1])
};

/// @brief A 2-qubit Pauli error that applies one of the following errors, with
/// the probabilities specified as inputs. Possible errors: IX, IY, IZ, XI, XX,
/// XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, and ZZ.
class pauli2 : public kraus_channel {
public:
  /// @brief Number of parameters. The 15 parameters are the probability that
  /// each one of the above errors will happen. Only 1 of the 15 possible
  /// errors will happen (at most).
  constexpr static std::size_t num_parameters = 15;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 2;

  /// @brief Construct a 2-qubit Pauli error Kraus channel
  /// @param p Error probabilities for the 2-qubit Pauli operators. The length
  /// of this vector must be 15. Note that since the probability of II is not
  /// specified, it is implied to by 1 - sum(p). Therefore, maximal mixing
  /// occurs when sum(p) = 0.9375.
  pauli2(const std::vector<cudaq::real> &p) {
    if (p.size() != num_parameters)
      throw std::runtime_error(
          "Invalid number of elements to pauli2 constructor. Must be 15.");
    cudaq::real sum = 0;
    for (auto pp : p) {
      if (pp < 0)
        throw std::runtime_error("Probabilities cannot be negative");
      sum += pp;
    }
    // This is just a first-level error check. Additional checks are done in
    // validateCompleteness.
    if (sum > static_cast<cudaq::real>(1.0 + 1e-6))
      throw std::runtime_error("Sum of pauli2 parameters is >1. Must be <= 1.");

    std::complex<cudaq::real> i{0, 1};
    cudaq::details::matrix_wrapper I({1, 0, 0, 1});
    cudaq::details::matrix_wrapper X({0, 1, 1, 0});
    cudaq::details::matrix_wrapper Y({0, -i, i, 0});
    cudaq::details::matrix_wrapper Z({1, 0, 0, -1});
    cudaq::real pii = std::max(static_cast<cudaq::real>(1.0 - sum),
                               static_cast<cudaq::real>(0));

    ops.reserve(16);
    // Use a lambda to avoid excessive line wrapping below
    auto define_op = [this](double _p,
                            const cudaq::details::matrix_wrapper &_m1,
                            const cudaq::details::matrix_wrapper &_m2) {
      ops.push_back(
          details::scale(std::sqrt(_p), details::kron(_m1, 2, 2, _m2, 2, 2)));
    };
    define_op(pii, I, I);
    define_op(p[0], I, X);
    define_op(p[1], I, Y);
    define_op(p[2], I, Z);
    define_op(p[3], X, I);
    define_op(p[4], X, X);
    define_op(p[5], X, Y);
    define_op(p[6], X, Z);
    define_op(p[7], Y, I);
    define_op(p[8], Y, X);
    define_op(p[9], Y, Y);
    define_op(p[10], Y, Z);
    define_op(p[11], Z, I);
    define_op(p[12], Z, X);
    define_op(p[13], Z, Y);
    define_op(p[14], Z, Z);

    this->parameters.reserve(p.size());
    for (auto pp : p)
      this->parameters.push_back(pp);
    noise_type = cudaq::noise_model_type::pauli2;
    validateCompleteness();
    generateUnitaryParameters();
  }
  REGISTER_KRAUS_CHANNEL(noise_model_strings[(int)noise_model_type::pauli2])
};

/// @brief depolarization1 is the same as depolarization_channel
class depolarization1 : public depolarization_channel {
public:
  depolarization1(const std::vector<cudaq::real> &p)
      : depolarization_channel(p) {
    noise_type = noise_model_type::depolarization1;
  }
  depolarization1(const real probability)
      : depolarization_channel(probability) {
    noise_type = noise_model_type::depolarization1;
  }
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::depolarization1])
};

/// @brief A 2-qubit depolarization error that applies one of the following
/// errors. Possible errors: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX,
/// ZY, and ZZ.
class depolarization2 : public kraus_channel {
public:
  /// @brief Number of parameters. The 1 parameter is the probability that each
  /// one of the 15 error possibilities list above will occur. Only 1 of the 15
  /// possible errors will happen (at most).
  constexpr static std::size_t num_parameters = 1;
  /// @brief Number of targets
  constexpr static std::size_t num_targets = 2;
  depolarization2(const std::vector<cudaq::real> p) : kraus_channel() {
    auto three = static_cast<cudaq::real>(3.);
    auto negOne = static_cast<cudaq::real>(-1.);
    auto probability = p[0];

    std::vector<std::vector<cudaq::complex>> singleQubitKraus = {
        {std::sqrt(1 - probability), 0, 0, std::sqrt(1 - probability)},
        {0, std::sqrt(probability / three), std::sqrt(probability / three), 0},
        {0, cudaq::complex{0, negOne * std::sqrt(probability / three)},
         cudaq::complex{0, std::sqrt(probability / three)}, 0},
        {std::sqrt(probability / three), 0, 0,
         negOne * std::sqrt(probability / three)}};

    // Generate 2-qubit Kraus operators
    ops.reserve(singleQubitKraus.size() * singleQubitKraus.size());
    for (const auto &k1 : singleQubitKraus) {
      for (const auto &k2 : singleQubitKraus) {
        ops.push_back(details::kron(k1, 2, 2, k2, 2, 2));
      }
    }
    this->parameters.push_back(probability);
    noise_type = cudaq::noise_model_type::depolarization2;
    validateCompleteness();
    generateUnitaryParameters();
  }

  /// @brief Construct a two qubit Kraus channel that applies a depolarization
  /// channel on either qubit independently.
  ///
  /// @param probability The probability of any depolarizing error happening in
  /// the 2 qubits. (Setting this to 1.0 ensures that "II" cannot happen;
  /// maximal mixing occurs at p = 0.9375.)
  depolarization2(const real probability)
      : depolarization2(std::vector<cudaq::real>{probability}) {}
  REGISTER_KRAUS_CHANNEL(
      noise_model_strings[(int)noise_model_type::depolarization2])
};

} // namespace cudaq
