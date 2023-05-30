/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <optional>

#include "armadillo"

#undef ARMA_USE_LAPACK

#include "ensmallen.hpp"

#include "ensmallen.h"

namespace {
class FunctionAdaptor {
protected:
  cudaq::optimizable_function &optFunction;

public:
  FunctionAdaptor(cudaq::optimizable_function &optF) : optFunction(optF) {}
  std::size_t NumFunctions() const { return 1; }
  void Shuffle() {}

  double Evaluate(const arma::mat &x) {
    std::vector<double> dummyGrad;
    auto xVec = arma::conv_to<std::vector<double>>::from(x);
    return optFunction(xVec, dummyGrad);
  }

  double Evaluate(const arma::mat &coordinates, const size_t begin,
                  const size_t batchSize) {
    return Evaluate(coordinates);
  }

  double EvaluateWithGradient(const arma::mat &x, arma::mat &dx) {
    std::vector<double> gradVec(dx.size());
    auto xVec = arma::conv_to<std::vector<double>>::from(x);
    auto e = optFunction(xVec, gradVec);
    for (std::size_t i = 0; i < dx.size(); i++) {
      dx(i) = gradVec[i];
    }
    return e;
  }

  double EvaluateWithGradient(const arma::mat &x, std::size_t bb, arma::mat &dx,
                              std::size_t b) {
    return EvaluateWithGradient(x, dx);
  }
};
} // namespace
namespace cudaq::optimizers {

void BaseEnsmallen::validate(optimizable_function &optFunction) {
  if (!optFunction.providesGradients() && requiresGradients())
    throw std::invalid_argument(
        R"#(Provided optimization function has invalid signature. 
        This optimizer requires gradients.
        Use signature double(const std::vector<double>& x, std::vector<double>& grad_x).)#");
}

optimization_result lbfgs::optimize(const int dim,
                                    optimizable_function &&opt_function) {
  validate(opt_function);

  FunctionAdaptor adaptor(opt_function);
  std::vector<double> x = initial_parameters.value_or(std::vector<double>(dim));
  arma::mat initCoords(x);

  auto localStepSize = step_size.value_or(1e-2);
  auto maxEval = max_eval.value_or(std::numeric_limits<std::size_t>::max());

  ens::L_BFGS lbfgs;
  lbfgs.MinStep() = localStepSize;
  lbfgs.MaxIterations() = maxEval;

  auto results = lbfgs.Optimize(adaptor, initCoords);
  return std::make_tuple(results,
                         arma::conv_to<std::vector<double>>::from(initCoords));
}

optimization_result adam::optimize(const int dim,
                                   optimizable_function &&opt_function) {
  validate(opt_function);

  FunctionAdaptor adaptor(opt_function);
  std::vector<double> x = initial_parameters.value_or(std::vector<double>(dim));
  arma::mat initCoords(x);

  auto localFtol = f_tol.value_or(1e-4);
  auto batchSize = batch_size.value_or(1);
  auto localStepSize = step_size.value_or(0.01);
  auto localBeta1 = beta1.value_or(0.9);
  auto localBeta2 = beta2.value_or(.999);
  auto localEps = eps.value_or(1e-8);
  auto maxEval = max_eval.value_or(std::numeric_limits<std::size_t>::max());

  ens::Adam adam(localStepSize, batchSize, localBeta1, localBeta2, localEps,
                 maxEval, localFtol, false, false, false);

  auto results = adam.Optimize(adaptor, initCoords);
  return std::make_tuple(results,
                         arma::conv_to<std::vector<double>>::from(initCoords));
}

optimization_result
gradient_descent::optimize(const int dim, optimizable_function &&opt_function) {
  validate(opt_function);

  FunctionAdaptor adaptor(opt_function);
  std::vector<double> x = initial_parameters.value_or(std::vector<double>(dim));
  arma::mat initCoords(x);

  auto localFtol = f_tol.value_or(1e-4);
  auto localStepSize = step_size.value_or(.01);
  auto maxEval = max_eval.value_or(std::numeric_limits<std::size_t>::max());

  ens::GradientDescent gd(localStepSize, maxEval, localFtol);

  auto results = gd.Optimize(adaptor, initCoords);
  return std::make_tuple(results,
                         arma::conv_to<std::vector<double>>::from(initCoords));
}

optimization_result sgd::optimize(const int dim,
                                  optimizable_function &&opt_function) {
  validate(opt_function);

  FunctionAdaptor adaptor(opt_function);
  std::vector<double> x = initial_parameters.value_or(std::vector<double>(dim));
  arma::mat initCoords(x);

  auto batchSize = batch_size.value_or(1);
  auto localFtol = f_tol.value_or(1e-4);
  auto localStepSize = step_size.value_or(.01);
  auto maxEval = max_eval.value_or(std::numeric_limits<std::size_t>::max());

  ens::VanillaUpdate vanillaUpdate;
  ens::StandardSGD sgd(localStepSize, batchSize, maxEval, localFtol, false,
                       vanillaUpdate, ens::NoDecay(), false, false);

  auto results = sgd.Optimize(adaptor, initCoords);
  return std::make_tuple(results,
                         arma::conv_to<std::vector<double>>::from(initCoords));
}

optimization_result spsa::optimize(const int dim,
                                   optimizable_function &&opt_function) {
  validate(opt_function);

  FunctionAdaptor adaptor(opt_function);
  std::vector<double> x = initial_parameters.value_or(std::vector<double>(dim));
  arma::mat initCoords(x);

  auto localFtol = f_tol.value_or(1e-4);
  auto localStepSize = step_size.value_or(0.16);
  auto localAlpha = alpha.value_or(0.602);
  auto localGamma = gamma.value_or(.101);
  auto localEvalStepSize = eval_step_size.value_or(.3);
  auto maxEval = max_eval.value_or(std::numeric_limits<std::size_t>::max());

  ens::SPSA spsa(localAlpha, localGamma, localStepSize, localEvalStepSize,
                 maxEval, localFtol);

  auto results = spsa.Optimize(adaptor, initCoords);
  return std::make_tuple(results,
                         arma::conv_to<std::vector<double>>::from(initCoords));
}

} // namespace cudaq::optimizers
