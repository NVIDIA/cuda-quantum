/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "common/JsonConvert.h"
#include "cudaq/algorithms/gradients/central_difference.h"
#include "cudaq/algorithms/gradients/forward_difference.h"
#include "cudaq/algorithms/gradients/parameter_shift.h"
#include "cudaq/algorithms/optimizers/ensmallen/ensmallen.h"
#include "cudaq/algorithms/optimizers/nlopt/nlopt.h"
#include "py_optimizer.h"
#include "py_utils.h"

namespace cudaq {

/// @brief Bind the `cudaq::optimization_result` typedef.
void bindOptimizationResult(py::module &mod) {
  py::class_<optimization_result>(mod, "OptimizationResult");
}

void bindGradientStrategies(py::module &mod) {
  // Binding under the `cudaq.gradients` namespace in python.
  auto gradients_submodule = mod.def_submodule("gradients");
  // Have to bind the parent class, `cudaq::gradient`, to allow
  // for the passing of arbitrary `cudaq::gradients::` around.
  // Note: this class lives under `cudaq.gradients.gradient`
  // in python.
  py::class_<gradient>(gradients_submodule, "gradient");
  // Gradient strategies derive from the `cudaq::gradient` class.
  py::class_<gradients::central_difference, gradient>(gradients_submodule,
                                                      "CentralDifference")
      .def(py::init<>())
      .def(
          "to_json",
          [](const gradients::central_difference &p) { return json(p).dump(); },
          "Convert gradient to JSON string")
      .def_static(
          "from_json",
          [](const std::string &j) {
            gradients::central_difference p;
            from_json(json::parse(j), p);
            return p;
          },
          "Convert JSON string to gradient")
      .def(
          "compute",
          [](cudaq::gradient &grad, const std::vector<double> &x,
             py::function &func, double funcAtX) {
            auto function =
                func.cast<std::function<double(std::vector<double>)>>();
            return grad.compute(x, function, funcAtX);
          },
          py::arg("parameter_vector"), py::arg("function"), py::arg("funcAtX"),
          "Compute the gradient of the provided `parameter_vector` with "
          "respect to "
          "its loss function, using the `CentralDifference` method.\n");
  py::class_<gradients::forward_difference, gradient>(gradients_submodule,
                                                      "ForwardDifference")
      .def(py::init<>())
      .def(
          "to_json",
          [](const gradients::forward_difference &p) { return json(p).dump(); },
          "Convert gradient to JSON string")
      .def_static(
          "from_json",
          [](const std::string &j) {
            gradients::forward_difference p;
            from_json(json::parse(j), p);
            return p;
          },
          "Convert JSON string to gradient")
      .def(
          "compute",
          [](cudaq::gradient &grad, const std::vector<double> &x,
             py::function &func, double funcAtX) {
            auto function =
                func.cast<std::function<double(std::vector<double>)>>();
            return grad.compute(x, function, funcAtX);
          },
          py::arg("parameter_vector"), py::arg("function"), py::arg("funcAtX"),
          "Compute the gradient of the provided `parameter_vector` with "
          "respect to "
          "its loss function, using the `ForwardDifference` method.\n");
  py::class_<gradients::parameter_shift, gradient>(gradients_submodule,
                                                   "ParameterShift")
      .def(py::init<>())
      .def(
          "to_json",
          [](const gradients::parameter_shift &p) { return json(p).dump(); },
          "Convert gradient to JSON string")
      .def_static(
          "from_json",
          [](const std::string &j) {
            gradients::parameter_shift p;
            from_json(json::parse(j), p);
            return p;
          },
          "Convert JSON string to gradient")
      .def(
          "compute",
          [](cudaq::gradient &grad, const std::vector<double> &x,
             py::function &func, double funcAtX) {
            auto function =
                func.cast<std::function<double(std::vector<double>)>>();
            return grad.compute(x, function, funcAtX);
          },
          py::arg("parameter_vector"), py::arg("function"), py::arg("funcAtX"),
          "Compute the gradient of the provided `parameter_vector` with "
          "respect to "
          "its loss function, using the `ParameterShift` method.\n");
}

/// @brief Add the requested optimization routine as a class
/// under the `cudaq.optimizers` sub-module namespace.
/// Can now define its member functions on
/// that submodule.
template <typename OptimizerT>
py::class_<OptimizerT> addPyOptimizer(py::module &mod, std::string &&name) {
  return py::class_<OptimizerT, optimizer>(mod, name.c_str())
      .def(py::init<>())
      .def(
          "to_json", [](const OptimizerT &p) { return json(p).dump(); },
          "Convert optimizer to JSON string")
      .def_static(
          "from_json",
          [](const std::string &j) {
            OptimizerT p;
            from_json(json::parse(j), p);
            return p;
          },
          "Convert JSON string to optimizer")
      .def_readwrite("max_iterations", &OptimizerT::max_eval, R"doc(
          int: Maximum number of optimizer iterations (default: unlimited).

          Sets an upper bound on the number of function evaluations or iterations 
          the optimizer will perform. If not set, the optimizer may run until 
          convergence or until another stopping criterion is met.
          )doc")
      .def_readwrite("initial_parameters", &OptimizerT::initial_parameters,
                     R"doc(
          list[float]: Initial values for the optimization parameters (optional).

          Provides a starting point for the optimization. If not specified, the 
          optimizer typically initializes parameters to zeros. Good initial 
          parameter values can significantly improve convergence speed and help 
          avoid poor local minima. The length must match the problem dimension.

          Example:
              .. code-block:: python

                  optimizer.initial_parameters = [0.5, -0.3, 1.2]
          )doc")
      .def_readwrite("lower_bounds", &OptimizerT::lower_bounds, R"doc(
          list[float]: Lower bounds for optimization parameters (optional).

          Constrains the search space by specifying minimum allowed values for 
          each parameter. When specified, the length must match the 
          problem dimension.

          Example:
              .. code-block:: python

                  optimizer.lower_bounds = [-2.0, -2.0]  # For 2D problem
          )doc")
      .def_readwrite("upper_bounds", &OptimizerT::upper_bounds, R"doc(
          list[float]: Upper bounds for optimization parameters (optional).

          Constrains the search space by specifying maximum allowed values for 
          each parameter. When specified, the length must match the 
          problem dimension.

          Example:
              .. code-block:: python

                  optimizer.upper_bounds = [2.0, 2.0]  # For 2D problem
          )doc")
      .def("requires_gradients", &OptimizerT::requiresGradients, R"doc(
          Check whether this optimizer requires gradient information.

          Returns:
              bool: True if the optimizer requires gradients (objective function 
                  must return (value, gradient)), False if gradient-free (objective 
                  function only needs to return value).

          Example:
              .. code-block:: python

                  optimizer = cudaq.optimizers.Adam()
                  if optimizer.requires_gradients():
                      print("Need to provide gradients")
          )doc")
      .def(
          "optimize",
          [](OptimizerT &opt, const int dim, py::function &func) {
            return opt.optimize(dim, [&](std::vector<double> x,
                                         std::vector<double> &grad) {
              // Call the function.
              auto ret = func(x);
              // Does it return a tuple?
              auto isTupleReturn = py::isinstance<py::tuple>(ret);
              // If we don't need gradients, and it does, just grab the value
              // and return.
              if (!opt.requiresGradients() && isTupleReturn)
                return ret.cast<py::tuple>()[0].cast<double>();
              // If we don't need gradients and it doesn't return tuple, then
              // just pass what we got.
              if (!opt.requiresGradients() && !isTupleReturn)
                return ret.cast<double>();

              // Throw an error if we need gradients and they weren't provided.
              if (opt.requiresGradients() && !isTupleReturn)
                throw std::runtime_error(
                    "Invalid return type on objective function, must return "
                    "(float, list[float]) for gradient-based optimizers");

              // If here, we require gradients, and the signature is right.
              auto tuple = ret.cast<py::tuple>();
              auto val = tuple[0];
              auto gradIn = tuple[1].cast<py::list>();
              for (std::size_t i = 0; i < gradIn.size(); i++)
                grad[i] = gradIn[i].cast<double>();

              return val.cast<double>();
            });
          },
          py::arg("dimensions"), py::arg("function"), R"doc(
Run the optimization procedure.

Args:
    dimensions (int): The number of parameters to optimize.
    function: The objective function to minimize. The signature 
        depends on whether the optimizer requires gradients:

        - For gradient-free optimizers: ``f(x: list[float]) -> float``
        - For gradient-based optimizers: ``f(x: list[float]) -> tuple[float, list[float]]``
          where the tuple contains (objective_value, gradient_vector)

Returns:
    tuple[float, list[float]]: A tuple containing (optimal_value, optimal_parameters)

Example:
    .. code-block:: python

        import cudaq

        # Gradient-free optimization
        optimizer = cudaq.optimizers.COBYLA()
        def objective(x):
            return x[0]**2 + x[1]**2

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective)

        # Gradient-based optimization
        optimizer = cudaq.optimizers.Adam()
        def objective_with_grad(x):
            value = x[0]**2 + x[1]**2
            gradient = [2*x[0], 2*x[1]]
            return (value, gradient)

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective_with_grad)
)doc");
}

void bindOptimizers(py::module &mod) {
  // Binding the `cudaq::optimizers` class to `_pycudaq` as a submodule
  // so it's accessible directly in the cudaq namespace.
  auto optimizers_submodule = mod.def_submodule("optimizers");
  py::class_<optimizer>(optimizers_submodule, "optimizer");

  addPyOptimizer<optimizers::cobyla>(optimizers_submodule, "COBYLA")
      .def(py::init<>(), R"doc(
Constrained Optimization BY Linear Approximations (COBYLA).

COBYLA is a gradient-free derivative-free optimization algorithm that uses 
linear approximations to handle constrained optimization problems.

This optimizer does not require gradients from the objective function.

Example:
    .. code-block:: python

        import cudaq

        optimizer = cudaq.optimizers.COBYLA()
        optimizer.max_iterations = 100

        def objective(x):
            return x[0]**2 + x[1]**2

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective)
)doc");

  addPyOptimizer<optimizers::neldermead>(optimizers_submodule, "NelderMead")
      .def(py::init<>(), R"doc(
Nelder-Mead simplex optimization algorithm.

The Nelder-Mead method is a gradient-free simplex-based optimization algorithm 
that iteratively refines a simplex of trial solutions.

This optimizer does not require gradients from the objective function.

Example:
    .. code-block:: python

        import cudaq

        optimizer = cudaq.optimizers.NelderMead()
        optimizer.max_iterations = 100

        def objective(x):
            return x[0]**2 + x[1]**2

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective)
)doc");

  addPyOptimizer<optimizers::lbfgs>(optimizers_submodule, "LBFGS")
      .def(py::init<>(), R"doc(
Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.

L-BFGS is a quasi-Newton method that approximates the Hessian matrix using 
limited memory. It is efficient for large-scale optimization problems.

This optimizer requires gradients from the objective function.

Example:
    .. code-block:: python

        import cudaq

        optimizer = cudaq.optimizers.LBFGS()
        optimizer.max_iterations = 100

        def objective_with_grad(x):
            value = x[0]**2 + x[1]**2
            gradient = [2*x[0], 2*x[1]]
            return (value, gradient)

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective_with_grad)
)doc");

  addPyOptimizer<optimizers::gradient_descent>(optimizers_submodule,
                                               "GradientDescent")
      .def(py::init<>(), R"doc(
Basic gradient descent optimization algorithm.

Gradient descent iteratively moves in the direction of steepest descent 
(negative gradient) to find a local minimum.

This optimizer requires gradients from the objective function.

Example:
    .. code-block:: python

        import cudaq

        optimizer = cudaq.optimizers.GradientDescent()
        optimizer.max_iterations = 100
        optimizer.step_size = 0.01

        def objective_with_grad(x):
            value = x[0]**2 + x[1]**2
            gradient = [2*x[0], 2*x[1]]
            return (value, gradient)

        opt_val, opt_params = optimizer.optimize(dimensions=2, function=objective_with_grad)
)doc");

  // Have to bind extra optimizer parameters to the following manually:
  auto py_spsa = addPyOptimizer<optimizers::spsa>(optimizers_submodule, "SPSA")
                     .def(py::init<>(), R"doc(
Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

SPSA is a gradient-free optimization algorithm that uses simultaneous 
perturbations to approximate gradients. It is particularly useful for 
noisy objective functions and requires only two function evaluations 
per iteration regardless of the problem dimension.

This optimizer does not require gradients from the objective function.

Example:
    .. code-block:: python

        import cudaq

        # SPSA optimizer
        optimizer = cudaq.optimizers.SPSA()
        optimizer.max_iterations = 100
        optimizer.step_size = 0.3  # Evaluation step size
        optimizer.gamma = 0.101    # Scaling exponent

        # Objective function
        def objective(x):
            return (x[0] - 2.0)**2 + (x[1] + 1.0)**2

        # Run optimization
        optimal_value, optimal_params = optimizer.optimize(
            dimensions=2, 
            function=objective
        )
)doc");
  py_spsa.def_readwrite("gamma", &cudaq::optimizers::spsa::gamma, R"doc(
float: Scaling exponent for the step size schedule (default: 0.101).

Controls how the step size decreases over iterations. The step size at 
iteration k is proportional to (A + k + 1)^(-gamma), where A is a stability 
constant. Common values are in the range [0.1, 0.6].
)doc");
  py_spsa.def_readwrite("step_size", &cudaq::optimizers::spsa::eval_step_size,
                        R"doc(
float: Evaluation step size for gradient approximation (default: 0.3).

Controls the magnitude of perturbations used to approximate gradients.
Larger values provide coarser gradient estimates but may be more robust
to noise. Typical values range from 0.1 to 0.5.
)doc");

  auto py_adam = addPyOptimizer<optimizers::adam>(optimizers_submodule, "Adam")
                     .def(py::init<>(), R"doc(
Adaptive Moment Estimation (Adam) optimizer.

Adam is an adaptive learning rate optimization algorithm that computes 
individual learning rates for different parameters from estimates of first 
and second moments of the gradients. It combines ideas from RMSProp and 
momentum optimization.

This optimizer requires gradients from the objective function. The objective 
function must return a tuple of (value, gradient_vector).

Example:
    .. code-block:: python

        import cudaq

        # Adam optimizer
        optimizer = cudaq.optimizers.Adam()
        optimizer.max_iterations = 100
        optimizer.step_size = 0.01      # Learning rate
        optimizer.beta1 = 0.9           # First moment decay
        optimizer.beta2 = 0.999         # Second moment decay
        optimizer.epsilon = 1e-8        # Numerical stability
        optimizer.f_tol = 1e-4          # Convergence tolerance

        # Objective function with gradients
        def objective_with_grad(x):
            value = (x[0] - 2.0)**2 + (x[1] + 1.0)**2
            grad = [2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)]
            return (value, grad)

        # Run optimization
        optimal_value, optimal_params = optimizer.optimize(
            dimensions=2, 
            function=objective_with_grad
        )
)doc");
  py_adam.def_readwrite("batch_size", &cudaq::optimizers::adam::batch_size,
                        R"doc(
int: Number of samples per batch (default: 1).

For stochastic optimization, determines how many samples are used to 
compute each gradient estimate. Batch size of 1 corresponds to online 
learning. Larger batch sizes can provide more stable gradient estimates
but require more computation per iteration.
)doc");
  py_adam.def_readwrite("beta1", &cudaq::optimizers::adam::beta1, R"doc(
float: Exponential decay rate for the first moment estimates (default: 0.9).

Controls the exponential moving average of past gradients (momentum term).
Values are typically in the range [0.9, 0.999]. Higher values give more 
weight to past gradients, providing smoother updates but slower adaptation.
)doc");
  py_adam.def_readwrite("beta2", &cudaq::optimizers::adam::beta2, R"doc(
float: Exponential decay rate for the second moment estimates (default: 0.999).

Controls the exponential moving average of past squared gradients. Values 
are typically in the range [0.99, 0.9999]. Higher values provide more 
stable learning rates but slower adaptation to changing gradient magnitudes.
)doc");
  py_adam.def_readwrite("epsilon", &cudaq::optimizers::adam::eps, R"doc(
float: Small constant for numerical stability (default: 1e-8).

Added to the denominator to prevent division by zero when computing 
adaptive learning rates. Should be a small positive value, typically 
between 1e-8 and 1e-6.
)doc");
  py_adam.def_readwrite("step_size", &cudaq::optimizers::adam::step_size, R"doc(
float: Learning rate (step size) for parameter updates (default: 0.01).

Controls the magnitude of parameter updates at each iteration. Typical 
values range from 0.001 to 0.1. The effective learning rate is adapted 
per parameter based on gradient history. Start with 0.001 or 0.01 and 
adjust based on convergence behavior.
)doc");
  py_adam.def_readwrite("f_tol", &cudaq::optimizers::adam::f_tol, R"doc(
float: Convergence tolerance on the objective function value (default: 1e-4).

Optimization terminates when the change in objective function value between 
iterations falls below this threshold. Smaller values lead to tighter 
convergence but may require more iterations.
)doc");

  auto py_sgd = addPyOptimizer<optimizers::sgd>(optimizers_submodule, "SGD")
                    .def(py::init<>(), R"doc(
Stochastic Gradient Descent (SGD) optimizer.

SGD is a fundamental optimization algorithm that updates parameters by taking 
steps proportional to the negative of the gradient. It is particularly 
effective for large-scale optimization problems and can escape local minima 
through the noise inherent in stochastic gradient estimates.

This optimizer requires gradients from the objective function. The objective 
function must return a tuple of (value, gradient_vector).

Example:
    .. code-block:: python

        import cudaq

        # SGD optimizer
        optimizer = cudaq.optimizers.SGD()
        optimizer.step_size = 0.01      # Learning rate
        optimizer.batch_size = 1        # Stochastic mode
        optimizer.f_tol = 1e-4          # Convergence tolerance

        # Objective function with gradients
        def objective_with_grad(x):
            value = (x[0] - 2.0)**2 + (x[1] + 1.0)**2
            grad = [2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)]
            return (value, grad)

        # Run optimization
        optimal_value, optimal_params = optimizer.optimize(
            dimensions=2, 
            function=objective_with_grad
        )
)doc");
  py_sgd.def_readwrite("batch_size", &cudaq::optimizers::sgd::batch_size, R"doc(
int: Number of samples per batch (default: 1).

For stochastic optimization, determines how many samples are used to 
compute each gradient estimate. Batch size of 1 corresponds to true 
stochastic gradient descent. Larger batch sizes (mini-batch SGD) can 
provide more stable gradient estimates but require more computation 
per iteration.
)doc");
  py_sgd.def_readwrite("step_size", &cudaq::optimizers::sgd::step_size, R"doc(
float: Learning rate (step size) for parameter updates (default: 0.01).

Controls the magnitude of parameter updates at each iteration. The update 
rule is: x_new = x_old - step_size * gradient. Typical values range from 
0.001 to 0.1. Too large values can cause divergence, while too small values 
lead to slow convergence.
)doc");
  py_sgd.def_readwrite("f_tol", &cudaq::optimizers::sgd::f_tol, R"doc(
float: Convergence tolerance on the objective function value (default: 1e-4).

Optimization terminates when the change in objective function value between 
iterations falls below this threshold. Smaller values lead to tighter 
convergence but may require more iterations. Note that with stochastic 
gradients, convergence may be noisy.
)doc");
}

void bindOptimizerWrapper(py::module &mod) {
  bindOptimizationResult(mod);
  bindGradientStrategies(mod);
  bindOptimizers(mod);
}

} // namespace cudaq
