/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "ObserveResult.h"
#include "Registry.h"
#include "SampleResult.h"

#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {

/// @brief Raw output produced by a runnable kernel. Defined fully in
/// cudaq/algorithms/run/policy.h; forward-declared here to break the include
/// cycle between the run policy header and this header.
struct run_result;

namespace detail {

/// @brief The execution context of a server job.
// Depending on the type, we may process the return data from the server
// differently when propagating it back to the runtime.
enum class ExecutionContextType : int { other = 0, sample = 1, observe, run };

/// Provider-neutral extension point for reopening serialized asynchronous
/// jobs that are not served through the CUDA-Q REST helper interface.
class JobResultRetriever : public registry::RegisteredType<JobResultRetriever> {
public:
  using Job = std::pair<std::string, std::string>;

  virtual ~JobResultRetriever() = default;
  virtual sample_result
  retrieve(const std::vector<Job> &jobs,
           const std::map<std::string, std::string> &config,
           ExecutionContextType resultType) = 0;
};

/// @brief The future type models the expected result of a
/// CUDA-Q kernel execution under a specific execution context.
/// This type is returned from asynchronous execution calls. It
/// encapsulates the job-specific circuit execution identifiers, the
/// name of the QPU the job executed on, and any extra configuration
/// information needed to retrieve the results later from the server.
/// This type can be persisted to file and read in later to retrieve
/// execution results.
/// It can also wrap a std::future<T> as a same-process fast path. Such a future
/// can be persisted only when provider job metadata is present for reopening
/// it in another process.
class future {
public:
  using Job = std::pair<std::string, std::string>;

protected:
  /// @brief Vector of job ids that make up the execution
  /// that this future corresponds to.
  std::vector<Job> jobs;

  // std::vector<std::string> jobNames;

  /// @brief The name of the QPU this execution is targeting
  std::string qpuName;

  /// @brief Any pertinent server configuration this future
  /// will require to retrieve results at a later time.
  std::map<std::string, std::string> serverConfig;

  /// @brief
  std::future<sample_result> inFuture;
  bool wrapsFutureSampling = false;

  /// @brief Indicate the execution context of this call
  ExecutionContextType resultType = ExecutionContextType::sample;

  /// Observable required to reconstruct a persisted asynchronous observe
  /// result. This is provider-neutral result metadata.
  std::optional<spin_op> persistedSpinOp;

  /// @brief Raw output data, if any, that is being returned
  /// from the server. This is used for `run` calls.
  std::vector<char> *inFutureRawOutput = nullptr;

public:
  /// @brief The constructor
  future() = default;

  /// @brief move constructor
  future(future &&) = default;
  future(std::future<sample_result> &f) : inFuture(std::move(f)) {
    wrapsFutureSampling = true;
  }

  future(std::future<sample_result> &&f) : inFuture(std::move(f)) {
    wrapsFutureSampling = true;
  }

  /// Same-process fast path plus all data required to persist and reopen the
  /// provider job in a later process.
  future(std::future<sample_result> &&f, std::vector<Job> jobs,
         std::string qpuName, std::map<std::string, std::string> config,
         ExecutionContextType type)
      : jobs(std::move(jobs)), qpuName(std::move(qpuName)),
        serverConfig(std::move(config)), inFuture(std::move(f)),
        wrapsFutureSampling(true), resultType(type) {}

  /// @brief The constructor, takes all info required to
  /// be able to retrieve results at a later date, even after file persistence.
  future(std::vector<Job> &_jobs, std::string &qpuNameIn,
         std::map<std::string, std::string> &config)
      : jobs(_jobs), qpuName(qpuNameIn), serverConfig(config) {}

  future(std::vector<Job> &_jobs, std::string &qpuNameIn,
         std::map<std::string, std::string> &config, ExecutionContextType type,
         std::vector<char> *rawOutput = nullptr)
      : jobs(_jobs), qpuName(qpuNameIn), serverConfig(config), resultType(type),
        inFutureRawOutput(rawOutput) {}

  future &operator=(future &other);
  future &operator=(future &&other) noexcept;

  sample_result get();

  void setSpinOp(const spin_op &op) {
    persistedSpinOp = op;
    persistedSpinOp->canonicalize();
  }

  [[nodiscard]] const std::optional<spin_op> &getSpinOp() const {
    return persistedSpinOp;
  }

  friend std::ostream &operator<<(std::ostream &, future &);
  friend std::istream &operator>>(std::istream &, future &);

private:
  bool isObserve() const { return resultType == ExecutionContextType::observe; }
};

std::ostream &operator<<(std::ostream &os, future &f);
std::istream &operator>>(std::istream &os, future &f);
} // namespace detail

/// @brief the async_result type is a user facing, future-like
/// type that is returned from CUDA-Q public asynchronous
/// API functions. It wraps a detail::future type, which can
/// itself be constructed from a std::future or a collection of
/// data pertinent to remote QPU REST invocation.
template <typename T>
class async_result {
protected:
  /// @brief The future holding data required to get the results later.
  detail::future result;

  /// @brief A spin operator, used for observe future tasks
  std::optional<spin_op> spinOp;

  /// @brief Raw output storage used for asynchronous run tasks.
  std::shared_ptr<std::vector<char>> rawOutput;

public:
  async_result() = default;
  async_result(const spin_op *s) {
    if (s) {
      spinOp = *s;
      spinOp.value().canonicalize();
    }
  }
  async_result(detail::future &&f, const spin_op *op = nullptr)
      : result(std::move(f)) {
    if (op) {
      spinOp = *op;
      spinOp.value().canonicalize();
      result.setSpinOp(*spinOp);
    }
  }
  async_result(detail::future &&f,
               std::shared_ptr<std::vector<char>> rawOutputIn)
      : result(std::move(f)), rawOutput(std::move(rawOutputIn)) {}

  virtual ~async_result() = default;
  async_result(async_result &&) = default;
  async_result &operator=(async_result &&other) = default;

  /// @brief Return the asynchronously computed data, will
  /// wait until the data is ready.
  T get() {
    auto data = result.get();

    if constexpr (std::is_same_v<T, sample_result>)
      return data;

    if constexpr (std::is_same_v<T, observe_result>) {
      if (!spinOp)
        spinOp = result.getSpinOp();
      if (!spinOp)
        throw std::runtime_error(
            "Returning an observe_result requires a spin_op.");

      // Server-side observe backends return a single expectation on the
      // global register (e.g. Fermioniq / external custom QPU plugins).
      if (data.has_expectation())
        return observe_result(data.expectation(), *spinOp, data);

      auto checkRegName = spinOp->to_string();
      if (data.has_expectation(checkRegName))
        return observe_result(data.expectation(checkRegName), *spinOp, data);

      // this assumes we ran in shots mode.
      double sum = 0.0;
      for (const auto &term : spinOp.value()) {
        if (term.is_identity())
          // FIXME: simply taking real here is very unclean at best,
          // and might be wrong/hiding a user error that should cause a failure
          // at worst. It would be good to not store a general spin op for the
          // result, but instead store the term ids and the evaluated
          // (double-valued) coefficient. Similarly, evaluate would fail if
          // the operator was parameterized. In general, both parameters, and
          // complex coefficients are valid for a spin-op term.
          // The code here (and in all other places that do something similar)
          // will work perfectly fine as long as there is no user error, but
          // the passed observable should really be validated properly and not
          // processed here as is making assumptions about correctness.
          sum += term.evaluate_coefficient().real();
        else
          sum += data.expectation(term.get_term_id()) *
                 term.evaluate_coefficient().real();
      }
      return observe_result(sum, *spinOp, data);
    }

    if constexpr (std::is_same_v<T, run_result>) {
      if (!rawOutput)
        throw std::runtime_error(
            "Returning a run_result requires raw output storage.");
      return {std::string(rawOutput->begin(), rawOutput->end())};
    }

    return T();
  }

  template <typename U>
  friend std::ostream &operator<<(std::ostream &, async_result<U> &);

  template <typename U>
  friend std::istream &operator>>(std::istream &, async_result<U> &);
};

template <typename T>
std::ostream &operator<<(std::ostream &os, async_result<T> &ar) {
  return os << ar.result;
}

template <typename T>
std::istream &operator>>(std::istream &is, async_result<T> &ar) {
  return is >> ar.result;
}

/// @brief Return type for asynchronous observation.
using async_observe_result = async_result<observe_result>;

/// @brief Return type for asynchronous sampling.
using async_sample_result = async_result<sample_result>;

/// @brief Return type for asynchronous runnable kernel execution.
using async_run_result = async_result<run_result>;

/// @brief Wrapper for a policy to return an async_result.
template <typename InnerPolicy>
struct async_policy_wrapper {
  InnerPolicy inner;
  using result_type = async_result<typename InnerPolicy::result_type>;
};

} // namespace cudaq
