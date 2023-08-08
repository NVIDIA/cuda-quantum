/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <optional>

#include "nlopt.hpp"

#include "nlopt.h"

namespace cudaq::optimizers {

struct ExtraNLOptData {
  std::function<double(const std::vector<double> &, std::vector<double> &)> f;
};

double nlopt_wrapper_call(const std::vector<double> &x,
                          std::vector<double> &grad, void *extra) {
  auto e = reinterpret_cast<ExtraNLOptData *>(extra);
  return e->f(x, grad);
}

#define CUDAQ_NLOPT_ALGORITHM_IMPL(CLASSNAME, ALGORITHM_NAME)                  \
  CLASSNAME::CLASSNAME() {}                                                    \
  CLASSNAME::~CLASSNAME() {}                                                   \
  optimization_result CLASSNAME::optimize(                                     \
      const int dim, optimizable_function &&opt_function) {                    \
    if (!opt_function.providesGradients() && requiresGradients()) {            \
      throw std::invalid_argument(                                             \
          "\nProvided optimization function has invalid signature.\nThis "     \
          "optimizer requires gradients.\nUse signature double(const "         \
          "std::vector<double>& x, std::vector<double>& grad_x).\n");          \
    }                                                                          \
    ::nlopt::algorithm algo = ::nlopt::algorithm::ALGORITHM_NAME;              \
    double tol = f_tol.value_or(1e-6);                                         \
    int maxeval = max_eval.value_or(std::numeric_limits<int>::max());          \
    std::vector<double> x =                                                    \
        initial_parameters.value_or(std::vector<double>(dim));                 \
    std::vector<double> lowerBounds =                                          \
        lower_bounds.value_or(std::vector<double>(dim, -M_PI));                \
    std::vector<double> upperBounds =                                          \
        upper_bounds.value_or(std::vector<double>(dim, M_PI));                 \
    if ((int)lowerBounds.size() != dim || (int)upperBounds.size() != dim) {    \
      throw std::invalid_argument(                                             \
          "\nThe dimensions of the bounds do not match the dimension "         \
          "of the initial_parameters.\nYou have provided " +                   \
          std::to_string(dim) + " initial_parameters, " +                      \
          std::to_string(lowerBounds.size()) + " lower_bounds and " +          \
          std::to_string(upperBounds.size()) + " upper_bounds.\n");            \
    }                                                                          \
    ExtraNLOptData data;                                                       \
    data.f = opt_function;                                                     \
    auto d = reinterpret_cast<void *>(&data);                                  \
    ::nlopt::opt _opt(algo, dim);                                              \
    _opt.set_min_objective(nlopt_wrapper_call, d);                             \
    _opt.set_lower_bounds(lowerBounds);                                        \
    _opt.set_upper_bounds(upperBounds);                                        \
    _opt.set_maxeval(maxeval);                                                 \
    _opt.set_ftol_rel(tol);                                                    \
    double optF;                                                               \
    try {                                                                      \
      _opt.optimize(x, optF);                                                  \
    } catch (const ::nlopt::roundoff_limited &e) {                             \
      throw std::runtime_error(                                                \
          "NLOpt error: round-off errors limited progress (" +                 \
          std::string(e.what()) +                                              \
          "). \nConsider changing the stopping criteria (e.g., reducing "      \
          "max_eval or increasing f_tol) and/or increase the number of "       \
          "measurement shots.");                                               \
    } catch (const ::nlopt::forced_stop &e) {                                  \
      throw std::runtime_error("NLOpt forced termination: " +                  \
                               std::string(e.what()));                         \
    } catch (const std::invalid_argument &e) {                                 \
      throw std::invalid_argument(                                             \
          "Invalid NLOpt arguments (e.g. lower bounds "                        \
          "are bigger than upper bounds): " +                                  \
          std::string(e.what()));                                              \
    } catch (const std::bad_alloc &e) {                                        \
      throw std::runtime_error("NLOpt ran out of memory.");                    \
    } catch (const std::runtime_error &e) {                                    \
      throw std::runtime_error("NLOpt runtime error: " +                       \
                               std::string(e.what()));                         \
    } catch (const std::exception &e) {                                        \
      throw std::runtime_error("Unknown NLOpt error: " +                       \
                               std::string(e.what()));                         \
    }                                                                          \
    return std::make_tuple(optF, x);                                           \
  }

CUDAQ_NLOPT_ALGORITHM_IMPL(cobyla, LN_COBYLA)
CUDAQ_NLOPT_ALGORITHM_IMPL(neldermead, LN_NELDERMEAD)

} // namespace cudaq::optimizers
