/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <regex>

#include "py_optimizer.h"

#include "common/JsonConvert.h"
#include "common/RemoteKernelExecutor.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq/algorithms/gradients/central_difference.h"
#include "cudaq/algorithms/gradients/forward_difference.h"
#include "cudaq/algorithms/gradients/parameter_shift.h"
#include "cudaq/algorithms/optimizers/ensmallen/ensmallen.h"
#include "cudaq/algorithms/optimizers/nlopt/nlopt.h"

namespace cudaq {

static std::string get_base64_encoded_var_dict() {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");

  py::dict serialized_dict;
  for (const auto item : py::globals()) {
    try {
      auto key = item.first;
      auto value = item.second;

      py::bytes serialized_value = pickle.attr("dumps")(value);
      serialized_dict[key] = serialized_value;
    } catch (const py::error_already_set &e) {
      // Uncomment the following lines for debug, but all this really means is
      // that we won't send this to the remote server.

      // std::cout << "Failed to pickle key: " + std::string(e.what())
      //           << std::endl;
    }
  }

  py::object inspect = py::module::import("inspect");
  std::vector<py::object> frame_vec;
  auto current_frame = inspect.attr("currentframe")();
  while (current_frame && !current_frame.is_none()) {
    frame_vec.push_back(current_frame);
    current_frame = current_frame.attr("f_back");
  }

  // Walk backwards through the call stack, which means we are going from
  // globals first to locals last. This ensures that the overwrites give
  // precedence to closest-to-locals.
  for (auto it = frame_vec.rbegin(); it != frame_vec.rend(); ++it) {
    py::dict f_locals = it->attr("f_locals");
    for (const auto item : f_locals) {
      try {
        auto key = item.first;
        auto value = item.second;

        py::bytes serialized_value = pickle.attr("dumps")(value);
        serialized_dict[key] = serialized_value;
      } catch (const py::error_already_set &e) {
        // Uncomment the following lines for debug, but all this really means is
        // that we won't send this to the remote server.

        // std::cout << "Failed to pickle key: " + std::string(e.what())
        //           << std::endl;
      }
    }
  }

  py::bytes serialized_code = pickle.attr("dumps")(serialized_dict);
  py::object encoded_dict = base64.attr("b64encode")(serialized_code);
  return encoded_dict.cast<std::string>();
}

// Find the minimum indent level for a set of lines in string and remove them
// from every line in the string.
static std::size_t strip_leading_whitespace(std::string &source_code) {
  std::size_t min_indent = std::numeric_limits<std::size_t>::max();

  // Traverse the lines to calculate min_indent.
  auto lines = cudaq::split(source_code, '\n');
  for (auto &line : lines) {
    std::size_t num_leading_whitespace = 0;
    bool non_space_found = false;
    for (auto c : line) {
      if (c == ' ' || c == '\t') {
        num_leading_whitespace++;
      } else {
        non_space_found = true;
        break;
      }
    }
    if (non_space_found)
      min_indent = std::min(min_indent, num_leading_whitespace);
    if (min_indent == 0)
      break;
  }

  // Now strip the leading indentation off the lines.
  source_code.clear();
  for (auto &line : lines)
    source_code += line.substr(std::min(line.size(), min_indent)) + '\n';

  return min_indent;
}

static std::string remove_comments(const std::string &source) {
  std::regex comment_regex(R"((\"[^\"]*\"|\'[^\']*\')|#.*)");
  std::string result;
  std::smatch match;
  std::string::const_iterator searchStart(source.cbegin());

  while (std::regex_search(searchStart, source.cend(), match, comment_regex)) {
    result += match.prefix().str();
    if (match[1].matched) {
      result += match.str();
    }
    searchStart = match.suffix().first;
  }
  result += std::string(searchStart, source.cend());
  return result;
}

std::string remove_print_statements(const std::string &source) {
  std::regex print_regex(R"((^\s*print\(.*\)\s*$))",
                         std::regex_constants::multiline);
  return std::regex_replace(source, print_regex, "");
}

static std::string get_source_code(const py::function &func) {
  // Get the source code
  py::object inspect = py::module::import("inspect");
  py::object source_code;
  try {
    source_code = inspect.attr("getsource")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  std::string source = source_code.cast<std::string>();
  strip_leading_whitespace(source);
  source = remove_comments(source);
  source = remove_print_statements(source);
  return remove_comments(source);
}

static std::string get_imports() {
  py::module sys = py::module::import("sys");
  py::dict sys_modules = sys.attr("modules");
  py::dict globals = py::globals();
  std::string imports_str;

  for (auto item : globals) {
    if (py::isinstance<py::module>(item.second)) {
      py::module mod = item.second.cast<py::module>();
      std::string alias = py::str(item.first);
      std::string name = py::str(mod.attr("__name__"));
      if (alias == name)
        imports_str += "import " + name + "\n";
      else
        imports_str += "import " + name + " as " + alias + "\n";
    }
  }
  return imports_str;
}

/// Form the SerializedCodeExecutionContext
static SerializedCodeExecutionContext
get_serialized_code(std::string &source_code) {
  SerializedCodeExecutionContext ctx;
  try {
    ctx.imports = get_imports();
    ctx.scoped_var_dict = get_base64_encoded_var_dict();
    ctx.source_code = source_code;
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to serialized data: " +
                             std::string(e.what()));
  }
  return ctx;
}

static std::string
get_required_raw_source_code(const int dim, const py::function &func,
                             const std::string &optimizer_var_name) {
  // Get source code and remove the leading whitespace
  std::string source_code = get_source_code(func);

  // Form the Python call to optimizer.optimize
  std::ostringstream os;
  auto obj_func_name = func.attr("__name__").cast<std::string>();
  os << "energy, params_at_energy = " << optimizer_var_name << ".optimize("
     << dim << ", " << obj_func_name << ")\n";
  auto function_call = os.str();

  // Return the combined code
  return source_code + "\n" + function_call;
}

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
      .def(py::pickle(
          [](const gradients::central_difference &p) { return json(p).dump(); },
          [](const std::string &data) {
            gradients::central_difference p;
            from_json(json::parse(data), p);
            return p;
          }))
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
      .def(py::pickle(
          [](const gradients::forward_difference &p) { return json(p).dump(); },
          [](const std::string &data) {
            gradients::forward_difference p;
            from_json(json::parse(data), p);
            return p;
          }))
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
      .def(py::pickle(
          [](const gradients::parameter_shift &p) { return json(p).dump(); },
          [](const std::string &data) {
            gradients::parameter_shift p;
            from_json(json::parse(data), p);
            return p;
          }))
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
      .def(py::pickle([](const OptimizerT &p) { return json(p).dump(); },
                      [](const std::string &data) {
                        OptimizerT p;
                        from_json(json::parse(data), p);
                        return p;
                      }))
      .def_readwrite("max_iterations", &OptimizerT::max_eval,
                     "Set the maximum number of optimizer iterations.")
      .def_readwrite("initial_parameters", &OptimizerT::initial_parameters,
                     "Set the initial parameter values for the optimization.")
      .def_readwrite(
          "lower_bounds", &OptimizerT::lower_bounds,
          "Set the lower value bound for the optimization parameters.")
      .def_readwrite(
          "upper_bounds", &OptimizerT::upper_bounds,
          "Set the upper value bound for the optimization parameters.")
      .def("requiresGradients", &OptimizerT::requiresGradients,
           "Returns whether the optimizer requires gradient.")
      .def(
          "optimize",
          [](OptimizerT &opt, const int dim, py::function &func) {
            auto &platform = cudaq::get_platform();
            if (platform.supports_remote_serialized_code() &&
                platform.num_qpus() == 1) {
              std::string optimizer_var_name = [&]() -> std::string {
                py::object inspect = py::module::import("inspect");
                // Search locals first, walking up the call stack
                auto current_frame = inspect.attr("currentframe")();
                while (current_frame && !current_frame.is_none()) {
                  py::dict f_locals = current_frame.attr("f_locals");
                  for (auto item : f_locals)
                    if (item.second.is(py::cast(&opt)))
                      return py::str(item.first);
                  current_frame = current_frame.attr("f_back");
                }
                // Search globals now
                current_frame = inspect.attr("currentframe")();
                py::dict f_globals = current_frame.attr("f_globals");
                for (auto item : f_globals)
                  if (item.second.is(py::cast(&opt)))
                    return py::str(item.first);
                throw std::runtime_error(
                    "Unable to find desired optimizer object in "
                    "global namespace. Aborting.");
              }();

              auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", 0);
              platform.set_exec_ctx(ctx.get());

              std::string combined_code =
                  get_required_raw_source_code(dim, func, optimizer_var_name);

              SerializedCodeExecutionContext serialized_code_execution_object =
                  get_serialized_code(combined_code);

              platform.launchSerializedCodeExecution(
                  func.attr("__name__").cast<std::string>(),
                  serialized_code_execution_object);

              platform.reset_exec_ctx();
              auto result = cudaq::optimization_result{};
              if (ctx->optResult)
                result = std::move(*ctx->optResult);
              return result;
            }

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
              // If we dont need gradients and it doesn't return tuple, then
              // just pass what we got.
              if (!opt.requiresGradients() && !isTupleReturn)
                return ret.cast<double>();

              // Throw an error if we need gradients and they weren't provided.
              if (opt.requiresGradients() && !isTupleReturn)
                throw std::runtime_error(
                    "Invalid return type on objective function, must return "
                    "(float,list[float]) for gradient-based optimizers");

              // If here, we require gradients, and the signature is right.
              auto tuple = ret.cast<py::tuple>();
              auto val = tuple[0];
              auto gradIn = tuple[1].cast<py::list>();
              for (std::size_t i = 0; i < gradIn.size(); i++)
                grad[i] = gradIn[i].cast<double>();

              return val.cast<double>();
            });
          },
          py::arg("dimensions"), py::arg("function"),
          "Run `cudaq.optimize()` on the provided objective function.");
}

void bindOptimizers(py::module &mod) {
  // Binding the `cudaq::optimizers` class to `_pycudaq` as a submodule
  // so it's accessible directly in the cudaq namespace.
  auto optimizers_submodule = mod.def_submodule("optimizers");
  py::class_<optimizer>(optimizers_submodule, "optimizer");

  addPyOptimizer<optimizers::cobyla>(optimizers_submodule, "COBYLA");
  addPyOptimizer<optimizers::neldermead>(optimizers_submodule, "NelderMead");
  addPyOptimizer<optimizers::lbfgs>(optimizers_submodule, "LBFGS");
  addPyOptimizer<optimizers::gradient_descent>(optimizers_submodule,
                                               "GradientDescent");

  // Have to bind extra optimizer parameters to the following manually:
  auto py_spsa = addPyOptimizer<optimizers::spsa>(optimizers_submodule, "SPSA");
  py_spsa.def_readwrite("gamma", &cudaq::optimizers::spsa::gamma,
                        "Set the value of gamma for the spsa optimizer.");
  py_spsa.def_readwrite("step_size", &cudaq::optimizers::spsa::eval_step_size,
                        "Set the step size for the spsa optimizer.");

  auto py_adam = addPyOptimizer<optimizers::adam>(optimizers_submodule, "Adam");
  py_adam.def_readwrite("batch_size", &cudaq::optimizers::adam::batch_size, "");
  py_adam.def_readwrite("beta1", &cudaq::optimizers::adam::beta1, "");
  py_adam.def_readwrite("beta2", &cudaq::optimizers::adam::beta2, "");
  py_adam.def_readwrite("episodes", &cudaq::optimizers::adam::eps, "");

  auto py_sgd = addPyOptimizer<optimizers::sgd>(optimizers_submodule, "SGD");
  py_sgd.def_readwrite("batch_size", &cudaq::optimizers::sgd::batch_size, "");
}

void bindOptimizerWrapper(py::module &mod) {
  bindOptimizationResult(mod);
  bindGradientStrategies(mod);
  bindOptimizers(mod);
}

} // namespace cudaq
