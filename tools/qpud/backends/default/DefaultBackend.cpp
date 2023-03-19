/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "TargetBackend.h"
#include "common/ExecutionContext.h"
#include "cudaq/utils/registry.h"
#include <cudaq/spin_op.h>

// Instantiate the registry for all backends
LLVM_INSTANTIATE_REGISTRY(cudaq::TargetBackend::RegistryType);

using TuplePtr = int8_t *;
struct Array;

extern "C" {
void __quantum__rt__setExecutionContext(cudaq::ExecutionContext *);
void __quantum__rt__resetExecutionContext();
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t);
int64_t __quantum__rt__array_get_size_1d(Array *);
Array *__quantum__rt__array_create_1d(int, int64_t);
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits);
void __quantum__qis__measure__body(Array *, Array *);
}

using namespace cudaq;

namespace cudaq {
bool kernelHasConditionalFeedback(const std::string &);
}
namespace {

Array *spinToArray(cudaq::spin_op &op) {
  // How to pack the data:
  // add all term data as correct pointer to double for x,y,z,or I.
  // After each term add a pointer to real part of term coeff,
  // add imag part of coeff.
  // End the data array with the number of terms in the list
  // x0 y1 - y0 x1 would be
  // 1 3 coeff.real coeff.imag 3 1 coeff.real coeff.imag NTERMS
  auto n_qubits = op.n_qubits();
  auto n_terms = op.n_terms();

  auto arr = __quantum__rt__array_create_1d(
      sizeof(double), n_qubits * n_terms + 2 * n_terms + 1);
  auto data = op.get_bsf();

  for (std::size_t i = 0; i < n_terms; i++) {
    auto term_data = data[i];
    std::size_t row_size = n_qubits + 2;
    for (std::size_t j = 0; j < row_size; j++) {
      int8_t *ptr =
          __quantum__rt__array_get_element_ptr_1d(arr, i * row_size + j);
      auto ptr_el = reinterpret_cast<double *>(ptr);
      if (j == n_qubits) {
        *ptr_el = op.get_term_coefficient(i).real();
        continue;
      }
      if (j == n_qubits + 1) {
        *ptr_el = op.get_term_coefficient(i).imag();
        break;
      }

      if (term_data[j] && term_data[j + n_qubits]) {
        // Y term
        *ptr_el = 3.0; // new double(3);
      } else if (term_data[j]) {
        // X term
        *ptr_el = 1.0; // new double(1.0);
      } else if (term_data[j + n_qubits]) {
        // Z term
        *ptr_el = 2.0; // new double(2);
      } else {
        *ptr_el = 0.0; // new double(0);
      }
    }
  }

  int8_t *ptr = __quantum__rt__array_get_element_ptr_1d(
      arr, n_qubits * n_terms + 2 * n_terms);
  auto ptr_el = reinterpret_cast<double *>(ptr);
  *ptr_el = n_terms;
  // cached_internal_data_rep = arr;
  return arr;
}

double measure(cudaq::spin_op &term, ExecutionContext *ctx) {
  Array *term_arr = spinToArray(term);
  __quantum__qis__measure__body(term_arr, nullptr);
  auto exp = ctx->expectationValue;
  return exp.value();
}

// The DefaultBackend subclasses TargetBackend to provide
// thunk function execution for base, sampling, and observation
// by delegating directly to QIR calls. This type relies on
// libnvqir being provided at link time.
class DefaultBackend : public TargetBackend {
public:
  void initialize() override { return; }
  DynamicResult baseExecute(Kernel &thunk, void *kernelArgs,
                            bool isClientServer) override {
    return thunk(kernelArgs, isClientServer);
  }
  std::vector<std::size_t> sample(Kernel &thunk, std::size_t shots,
                                  void *kernelArgs) override {
    cudaq::ExecutionContext ctx("sample", shots);

    // First see if this Quake representation has conditional feedbackF
    auto quakeCode = thunk.getQuakeCode();
    ctx.hasConditionalsOnMeasureResults =
        !quakeCode.empty() &&
        quakeCode.find("qubitMeasurementFeedback = true") != std::string::npos;

    __quantum__rt__setExecutionContext(&ctx);

    sample_result counts;
    if (ctx.hasConditionalsOnMeasureResults) {
      // If it has conditionals, loop over individual circuit executions
      for (auto &i : cudaq::range(shots)) {
        // Run the kernel
        thunk(kernelArgs, /*isClientServer=*/false);
        // Reset the context and get the single measure result,
        // add it to the sample_result and clear the context result
        __quantum__rt__resetExecutionContext();
        counts += ctx.result;
        ctx.result.clear();
        // Reset the context for the next round
        if (i < (unsigned)shots)
          __quantum__rt__setExecutionContext(&ctx);
      }
      return counts.serialize();
    }

    // Just run the kernel, context will get the sampling results
    thunk(kernelArgs, /*isClientServer=*/false);
    __quantum__rt__resetExecutionContext();
    return ctx.result.serialize();
  }
  std::tuple<double, std::vector<std::size_t>>
  observe(Kernel &thunk, std::vector<double> &spin_op_data,
          const std::size_t shots, void *kernelArgs) override {

    ExecutionContext ctx("observe");
    ctx.shots = shots == 0 ? -1 : shots;

    __quantum__rt__setExecutionContext(&ctx);
    // default to never using CircuitSimulator::observe()
    ctx.canHandleObserve = false;

    auto n_terms = (int)spin_op_data.back();
    auto nQubits = (spin_op_data.size() - 2 * n_terms) / n_terms;
    cudaq::spin_op H(spin_op_data, nQubits);
    double sum = 0.0;
    thunk(kernelArgs, /*isClientServer=*/false);

    std::vector<ExecutionResult> results;
    for (std::size_t i = 0; i < H.n_terms(); i++) {
      auto term = H[i];
      if (!term.is_identity()) {
        auto exp = measure(term, &ctx);
        results.emplace_back(ctx.result.to_map(), term.to_string(false));
        sum += term.get_term_coefficient(0).real() * exp;
      } else {
        sum += term.get_term_coefficient(0).real();
      }
    }
    __quantum__rt__resetExecutionContext();
    sample_result counts(results);
    return std::make_tuple(sum, counts.serialize());
  }

  std::string genRandomString() {
    const int len = 32;
    static const char alphanum[] = "0123456789"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz";
    std::string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
      tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
  }

  std::tuple<std::string, std::string>
  sampleDetach(Kernel &thunk, std::size_t shots, void *kernelArgs) override {
    cudaq::ExecutionContext ctx("sample", shots);
    __quantum__rt__setExecutionContext(&ctx);
    thunk(kernelArgs, /*isClientServer=*/false);
    __quantum__rt__resetExecutionContext();
    auto jobId = genRandomString();
    detachedSampleResults.insert({jobId, ctx.result});
    return std::make_tuple(jobId,
                           std::string(thunk.name()) + std::string(".sample"));
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>>
  observeDetach(Kernel &thunk, std::vector<double> &spin_op_data,
                const std::size_t shots, void *kernelArgs) override {
    // Local declarations
    ExecutionContext ctx("observe");
    ctx.shots = shots == 0 ? -1 : shots;

    __quantum__rt__setExecutionContext(&ctx);
    auto n_terms = (int)spin_op_data.back();
    auto nQubits = (spin_op_data.size() - 2 * n_terms) / n_terms;

    cudaq::spin_op H(spin_op_data, nQubits);
    thunk(kernelArgs, /*isClientServer=*/false);

    std::vector<std::string> jobIds, jobNames;
    for (std::size_t i = 0; i < H.n_terms(); i++) {
      auto term = H[i];
      if (!term.is_identity()) {
        auto jobId = genRandomString();
        auto exp = measure(term, &ctx);
        detachedObserveResults.insert(
            {jobId, std::make_pair(exp, ctx.result.serialize())});
        jobIds.push_back(jobId);
        jobNames.push_back(term.to_string(false));
      }
    }
    __quantum__rt__resetExecutionContext();

    return std::make_tuple(jobIds, jobNames);
  }

  std::tuple<double, std::vector<std::size_t>>
  observeFromJobId(const std::string &jobId) override {
    return detachedObserveResults[jobId];
  }

  std::vector<std::size_t> sampleFromJobId(const std::string &jobId) override {
    return detachedSampleResults[jobId].serialize();
  }

  virtual ~DefaultBackend() = default;

protected:
  /// @brief For each detached sample task, store jobId -> results
  std::map<std::string, sample_result> detachedSampleResults;

  /// @brief For each detached observe task, store jobId -> tuple(ExpVal,
  /// Serialized Counts)
  std::map<std::string, std::tuple<double, std::vector<std::size_t>>>
      detachedObserveResults;
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::TargetBackend, DefaultBackend, default);
