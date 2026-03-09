/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SampleResult.h"
#include "cudaq/operators.h"

#include <cassert>

namespace cudaq {

/// @brief The observe_result encapsulates all data generated from a
/// cudaq"::"observe call. This includes all measurement counts from the
/// execution of each ansatz + measure circuit, and the global expected
/// value of the spin_op passed to observe.
class observe_result {
protected:
  // The global expectation for the given spinOp
  double expVal = 0.0;

  // The spin_op measured
  spin_op spinOp;

  // Contains ExecutionResults for all terms measured
  sample_result data;

public:
  observe_result() = default;

  /// @brief Constructor, takes the precomputed expectation value for
  /// <psi(x) | H | psi(x)> for the given spin_op.
  observe_result(double e, const spin_op &H) : expVal(e), spinOp(H) {
    assert(cudaq::spin_op::canonicalize(spinOp) == spinOp);
  }

  /// @brief Constructor, takes the precomputed expectation value for
  /// <psi(x) | H | psi(x)> for the given spin_op. If this execution
  /// was shots based, also provide the sample_result data containing counts
  /// for each term in H.
  observe_result(double e, const spin_op &H, sample_result counts)
      : expVal(e), spinOp(H), data(counts) {
    assert(cudaq::spin_op::canonicalize(spinOp) == spinOp);
  }

  /// @brief Return the raw counts data for all terms
  /// @return
  sample_result raw_data() { return data; };

  /// @brief Conversion operator for this `observe_result` to `double`. Simply
  /// returns the precomputed expectation value for the given spin_op. This
  /// enables one to ignore the fine-grain sample_result data, and explicitly
  /// request the expected value: double exp = cudaq"::"observe(...); as opposed
  /// to cudaq"::"observe_data data = cudaq::observe(...); auto exp =
  /// data.expectation();
  operator double() { return expVal; }

  /// @brief Return the expected value for the provided spin_op
  /// @return
  double expectation() { return expVal; }

  /// @brief Return the expectation value for a sub-term in the provided
  /// spin_op.
  double expectation(const spin_op_term &term) {
    auto termStr = cudaq::spin_op_term::canonicalize(term).get_term_id();
    return data.expectation(termStr);
  }

  [[deprecated("passing a spin_op to expectation is deprecated - please pass a "
               "spin_op_term instead")]] double
  expectation(const spin_op &op) {
    if (op.num_terms() != 1)
      throw std::runtime_error("expecting a spin op with exactly one term");
    auto termStr = cudaq::spin_op_term::canonicalize(*op.begin()).get_term_id();
    return data.expectation(termStr);
  }

  /// @brief Return the counts data for the given spin_op
  sample_result counts(const spin_op_term &term) {
    auto termStr = cudaq::spin_op_term::canonicalize(term).get_term_id();
    auto counts = data.to_map(termStr);
    ExecutionResult result(counts);
    return sample_result(result);
  }

  [[deprecated("passing a spin_op to counts is deprecated - please pass a "
               "spin_op_term instead")]] sample_result
  counts(const spin_op &op) {
    if (op.num_terms() != 1)
      throw std::runtime_error("expecting a spin op with exactly one term");
    auto termStr = cudaq::spin_op_term::canonicalize(*op.begin()).get_term_id();
    auto counts = data.to_map(termStr);
    ExecutionResult result(counts);
    return sample_result(result);
  }

  /// @brief Return the coefficient of the identity term.
  /// Assumes there is at more one identity term.
  /// Returns 0 if no identity term exists.
  double id_coefficient() {
    for (const auto &term : spinOp)
      if (term.is_identity())
        return term.evaluate_coefficient().real();
    return 0.0;
  }

  spin_op get_spin() { return spinOp; }

  /// @brief Dump the counts data to standard out.
  void dump() { data.dump(); }
};

} // namespace cudaq
