/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {
/// Type definition for the mapping of observed qubit measurement bit strings
/// to the number of times they were observed.
using CountsDictionary = std::unordered_map<std::string, std::size_t>;

inline static const std::string GlobalRegisterName = "__global__";

/// The `ExecutionResult` models the result of a typical
/// quantum state sampling task. It will contain the
/// observed measurement bit strings and corresponding number
/// of times observed, as well as an expected value with
/// respect to the Z...Z operator.
struct ExecutionResult {
  // Measurements and times observed
  CountsDictionary counts;

  // <Z...Z> expected value
  std::optional<double> expectationValue = std::nullopt;

  /// Register name for the classical bits
  std::string registerName = GlobalRegisterName;

  /// @brief Sequential bit strings observed (not collated into a map)
  std::vector<std::string> sequentialData;

  /// @brief Serialize this sample result to a vector of integers.
  /// Encoding: 1st element is size of the register name N, then next N
  /// represent register name, next is the number of bitstrings M,
  /// then for each bit string a triple {string mapped to long, bit string
  /// length, count}
  /// @return
  std::vector<std::size_t> serialize();

  /// @brief Deserialize a vector of integers to a `ExecutionResult`
  /// @param data The data with encoding discussed in the brief for
  /// `serialize`.
  void deserialize(std::vector<std::size_t> &data);

  /// @brief Constructor
  ExecutionResult() = default;

  /// @brief Construct from a CountsDictionary, assumes registerName ==
  /// __global__
  /// @param c the counts
  ExecutionResult(CountsDictionary c);

  /// @brief Construct from a register name
  ExecutionResult(std::string name);

  /// @brief Construct from a precomputed expectation value
  ExecutionResult(double expVal);

  /// @brief Construct from a `CountsDictionary`, specify the register name
  /// @param c the counts
  /// @param name the register name
  ExecutionResult(CountsDictionary c, std::string name);
  ExecutionResult(CountsDictionary c, std::string name, double exp);

  /// @brief Construct from a `CountsDictionary` and expected value
  /// @param c The counts
  /// @param e The precomputed expected value
  ExecutionResult(CountsDictionary c, double e);

  /// @brief Copy constructor
  /// @param other
  ExecutionResult(const ExecutionResult &other);

  /// @brief Set this ExecutionResult equal to the provided one
  /// @param other
  /// @return
  ExecutionResult &operator=(ExecutionResult &other);

  /// @brief Return true if the given `ExecutionResult` is the same as this one.
  /// @param result
  /// @return
  bool operator==(const ExecutionResult &result) const;

  /// @brief Append the bitstring and count to this `ExecutionResult`
  /// @param bitString
  /// @param count
  void appendResult(std::string bitString, std::size_t count);

  std::vector<std::string> getSequentialData() { return sequentialData; }
};

/// @brief The sample_result abstraction wraps a set of `ExecutionResult`s for
/// a single quantum kernel execution under the sampling or observation
/// `ExecutionContext`. Each `ExecutionResult` is mapped to a register name,
/// with a default `ExecutionResult` with name __global__ representing the
/// observed measurement results holistically for the quantum kernel.
class sample_result {
private:
  /// @brief A mapping of register names to `ExecutionResult`s
  std::unordered_map<std::string, ExecutionResult> sampleResults;

  /// @brief Keep track of the total number of shots. We keep this
  /// here so we don't have to keep recomputing it.
  std::size_t totalShots = 0;

public:
  /// @brief Nullary constructor
  sample_result() = default;

  /// @brief The constructor, sets the __global__ sample result.
  /// @param result
  sample_result(ExecutionResult &result);

  /// @brief The constructor, appends all provided `ExecutionResult`s
  sample_result(std::vector<ExecutionResult> &results);

  /// @brief The constructor, takes a precomputed expectation value and
  /// stores it with the `__global__` `ExecutionResult`.
  sample_result(double preComputedExp, std::vector<ExecutionResult> &results);

  /// @brief Copy Constructor
  sample_result(const sample_result &);

  /// @brief The destructor
  ~sample_result() = default;

  /// @brief Return true if the `ExecutionResult` with the specified register
  /// name has a precomputed expectation value.
  bool
  has_expectation(const std::string_view registerName = GlobalRegisterName);

  /// @brief Add another `ExecutionResult` to this `sample_result`.
  /// @param result
  void append(ExecutionResult &result);

  /// @brief Return all register names. Can be used in tandem with
  /// sample_result::to_map(regName : string) to retrieve the counts
  /// for each register.
  /// @return
  std::vector<std::string> register_names();

  /// @brief Set this sample_result equal to the provided one
  /// @param counts
  /// @return
  sample_result &operator=(sample_result &counts);
  sample_result &operator=(const sample_result &counts);

  /// @brief Append all the data from other to this sample_result.
  /// Merge when necessary.
  /// @param other
  /// @return
  sample_result &operator+=(sample_result &other);

  /// @brief Serialize this sample_result. See
  /// `ExecutionResult::serialize()` documentation for information
  //. about the encoding.
  /// @return
  std::vector<std::size_t> serialize();

  /// @brief Create this sample_result from the serialized data.
  /// @param data
  void deserialize(std::vector<std::size_t> &data);

  /// @brief Return true if this sample_result is the same as the given one
  /// @param counts
  /// @return
  bool operator==(const sample_result &counts) const;

  /// @brief Return the expected value <Z...Z>
  /// @return
  double exp_val_z(const std::string_view registerName = GlobalRegisterName);

  /// @brief Return the probability of observing the given bit string
  /// @param bitString
  /// @return
  double probability(std::string_view bitString,
                     const std::string_view registerName = GlobalRegisterName);

  /// @brief Return the most probable bit string.
  /// @param registerName
  /// @return
  std::string
  most_probable(const std::string_view registerName = GlobalRegisterName);

  /// @brief Return the number of times the given bitstring was observed
  /// @param bitString
  /// @return
  std::size_t count(std::string_view bitString,
                    const std::string_view registerName = GlobalRegisterName);

  std::vector<std::string>
  sequential_data(const std::string_view registerName = GlobalRegisterName);

  /// @brief Return the number of observed bit strings
  /// @return
  std::size_t
  size(const std::string_view registerName = GlobalRegisterName) noexcept;

  /// @brief Dump this sample_result to standard out.
  void dump();

  /// @brief Dump this sample_result to the given output stream
  void dump(std::ostream &os);

  /// @brief Clear this sample_result.
  void clear();

  /// @brief Extract the ExecutionResults as a std::unordered<string, size_t>
  /// map.
  /// @param registerName
  /// @return
  CountsDictionary
  to_map(const std::string_view registerName = GlobalRegisterName);

  /// @brief Extract marginal counts, that is those counts for a subset
  /// of measured qubits
  /// @param marginalIndices The qubit indices as an `rvalue`
  /// @param registerName
  /// @return
  sample_result
  get_marginal(const std::vector<std::size_t> &&marginalIndices,
               const std::string_view registerName = GlobalRegisterName) {
    return get_marginal(marginalIndices);
  }

  /// @brief Extract marginal counts, that is those counts for a subset
  /// of measured qubits
  /// @param marginalIndices The qubit indices as an reference
  /// @param registerName
  /// @return
  sample_result
  get_marginal(const std::vector<std::size_t> &marginalIndices,
               const std::string_view registerName = GlobalRegisterName);

  /// @brief Range-based iterator begin function
  /// @return
  CountsDictionary::iterator begin();

  /// @brief Range-based iterator end function
  /// @return
  CountsDictionary::iterator end();

  /// @brief Range-based constant iterator begin function
  /// @return
  CountsDictionary::const_iterator cbegin() const;

  /// @brief Range-based constant iterator end function
  /// @return
  CountsDictionary::const_iterator cend() const;

  /// @brief Range-based constant iterator begin function
  /// @return
  CountsDictionary::const_iterator begin() const { return cbegin(); }

  /// @brief Range-based constant iterator end function
  /// @return
  CountsDictionary::const_iterator end() const { return cend(); }

  /// @brief Return true if the bit string has even parity
  /// @param bitString
  /// @return
  static bool has_even_parity(std::string_view bitString);
};

} // namespace cudaq
