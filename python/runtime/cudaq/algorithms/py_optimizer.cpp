/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nlohmann/json.hpp"
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_optimizer.h"

#include "common/SerializedCodeExecutionContext.h"
#include "common/RemoteKernelExecutor.h"
#include "cudaq/algorithms/gradients/central_difference.h"
#include "cudaq/algorithms/gradients/forward_difference.h"
#include "cudaq/algorithms/gradients/parameter_shift.h"
#include "cudaq/algorithms/optimizers/ensmallen/ensmallen.h"
#include "cudaq/algorithms/optimizers/nlopt/nlopt.h"
#include "llvm/Support/Base64.h"
#include "mlir/CAPI/IR.h"
#include "utils/LinkedLibraryHolder.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "utils/OpaqueArguments.h"

#include <fstream>
#include <iostream>

using json = nlohmann::json;
using namespace mlir;

namespace cudaq {

/// @brief Bind the `cudaq::optimization_result` typedef.
void bindOptimizationResult(py::module &mod) {
  py::class_<optimization_result>(mod, "OptimizationResult");
}

std::string get_base64_encoded_str(const std::string &source_code) {
  // Encode the serialized data to Base64
  return llvm::encodeBase64(source_code);
}

std::string get_base64_encoded_str(const py::dict &namespace_dict) {
  py::object json = py::module_::import("json");
  
  py::exec(R"(
    import json

    class CustomEncoder(json.JSONEncoder):
      def default(self, obj):
        try:
          return super().default(obj);
        except TypeError:
          return str(obj);
  )");
  py::object custom_encoder = py::globals()["CustomEncoder"];
  py::object json_string = json.attr("dumps")(namespace_dict, py::arg("cls")=custom_encoder);
  std::string json_cpp_string = json_string.cast<std::string>();
  return llvm::encodeBase64(json_cpp_string);
}

json serialize_data(std::string &source_code, py::dict &locals,
                    py::dict &globals) {
  std::string encoded_code_str = get_base64_encoded_str(source_code);
  std::string encoded_locals_str = get_base64_encoded_str(locals);
  std::string encoded_globals_str = get_base64_encoded_str(globals);

  json json_object;
  json_object["source_code"] = encoded_code_str;
  json_object["locals"] = encoded_locals_str;
  json_object["globals"] = encoded_globals_str;

  return json_object;
}

std::string extract_cudaq_target_parameter(const std::string &line) {
  std::string target_param;
  std::string search_str = "cudaq.set_target(";
  auto start_pos = line.find(search_str);

  if (start_pos != std::string::npos &&
      line.substr(0, start_pos).find("#") == std::string::npos) {
    start_pos += search_str.length();
    auto end_pos = line.find(")", start_pos);
    if (end_pos != std::string::npos) {
      target_param = line.substr(start_pos, end_pos - start_pos);

      target_param.erase(0, target_param.find_first_not_of(" \t\n\r\"'"));
      target_param.erase(target_param.find_last_not_of(" \t\n\r\"'") + 1);
    }
  }

  return target_param;
}

std::string read_file(const std::string &file_path) {
  std::ifstream file_handle(file_path);
  if (!file_handle) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  return std::string((std::istreambuf_iterator<char>(file_handle)),
                     std::istreambuf_iterator<char>());
}

std::string get_file_content(const py::object &inspect,
                             const py::function &func) {
  // Get the source file of the function
  py::object source_file;
  try {
    source_file = inspect.attr("getfile")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source file: " +
                             std::string(e.what()));
  }

  // Read the entire source file
  std::string file_path = source_file.cast<std::string>();
  std::string file_content = read_file(file_path);
  return file_content;
}

py::object get_source_code(const py::object &inspect,
                           const py::function &func) {
  // Get the source code
  py::object source_code;
  try {
    source_code = inspect.attr("getsource")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  return source_code;
}

py::object get_parent_frame_info(const py::object &inspect) {
  // Get the stack and retrieve the parent frame
  py::list stack = inspect.attr("stack")();
  // TODO: work on recursive frames
  if (py::len(stack) < 1) {
    throw std::runtime_error("Insufficient stack depth to find parent frame.");
  }

  return stack[0];
}

std::tuple<py::dict, py::dict>
get_locals_and_globals(const py::object &parent_frame_info) {
  py::dict locals;
  py::dict globals;
  try {
    locals = parent_frame_info.attr("frame").attr("f_locals");
    globals = parent_frame_info.attr("frame").attr("f_globals");
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get frame locals and globals: " +
                             std::string(e.what()));
  }

  return std::make_tuple(locals, globals);
}

py::object get_compiled_code(const std::string &combined_code,
                             const py::object &builtins) {
  py::object compiled_code;
  try {
    py::object compile = builtins.attr("compile");
    compiled_code = compile(combined_code, "<string>", "exec");
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to compile code: " +
                             std::string(e.what()));
  }

  return compiled_code;
}

SerializedCodeExecutionContext *get_serialized_code(std::string &source_code, py::dict &locals,
                         py::dict &globals) {
  // Serialize the source code, locals, and globals
  json serialized_data, empty_json;
  try {
    serialized_data = serialize_data(source_code, locals, globals);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to serialized data: " +
                             std::string(e.what()));
  }

  SerializedCodeExecutionContext *serializedCodeExecutionContext = new SerializedCodeExecutionContext(serialized_data["source_code"], serialized_data["locals"], serialized_data["globals"]);

  return serializedCodeExecutionContext;
}

std::tuple<std::string, std::string>
extract_func_call_and_cudaq_target(const std::string &file_content,
                                   const py::object &func) {
  // Split the file content into lines and process each line
  std::istringstream file_stream(file_content);
  std::string line;
  std::string func_call;
  std::string cudaq_target;
  while (std::getline(file_stream, line)) {
    if (line.find(func.attr("__name__").cast<std::string>()) !=
        std::string::npos) {
      func_call = line;
    }
    if (line.find("cudaq.set_target") != std::string::npos) {
      cudaq_target = extract_cudaq_target_parameter(line);
    }
  }

  if (func_call.empty()) {
    throw std::runtime_error("Failed to find function call in parent frame");
  }

  return std::make_tuple(func_call, cudaq_target);
}

std::string create_return_string(const std::string &func_call) {
  auto pos = func_call.find('=');

  if (pos != std::string::npos) {
    std::string str = func_call.substr(0, pos);
    return "return " + str.substr(0, str.find_last_not_of(" \t\n\r") + 1) + ";";
  }

  return "";
}

void call_rest_api(SerializedCodeExecutionContext *serializeCodeExecutionObject, const py::dict &locals, const py::dict &globals) {
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client 
    = cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest");
  
  py::object kernel = locals["kernel"];
  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  auto &platform = get_platform();
  auto *ctx = platform.get_exec_ctx();
  using namespace mlir;

  ModuleOp mod = unwrap(kernelMod);
  auto *mlirContext = mod->getContext();

  std::cout << "Extracting required parameters..." << std::endl;
  std::cout << "Kernel: " << kernel << std::endl;

  std::cout << "Global namespace:\n" << std::endl;
  std::cout << py::globals() << std::endl;
  // auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);

  py::list attributes = kernel.attr("__dict__").attr("items")();
  for (auto item : attributes) {
    py::tuple item_arr = item.cast<py::tuple>();
    auto key = item_arr[0];
    auto value = item_arr[1];
    std::cout << key.cast<std::string>() << ": " << py::str(value).cast<std::string>() << std::endl;
  }

  std::cout << "Making a REST request..." << std::endl;
  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(*mlirContext, *ctx, *serializeCodeExecutionObject, "", kernelName,
                              nullptr, nullptr, 0, &errorMsg);
  
  if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
}

std::string trim(const std::string &str) {
  std::stringstream ss;
  ss << str;
  std::string trimmed;
  ss >> trimmed;
  return trimmed;
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
      .def(
          "optimize",
          // objective_function
          // use C++ function
          [](OptimizerT &opt, const int dim, py::function &func) {
            // write a function to detect if cudaq.set_target has been used and
            // fetch its value

            py::object inspect = py::module::import("inspect");

            // Get source file content
            std::string file_content = get_file_content(inspect, func);

            // Extract function call and cudaq target from the file content
            auto [func_call, cudaq_target] =
                extract_func_call_and_cudaq_target(file_content, func);

            // Run this only if the cudaq target is set to nvqc
            if (cudaq_target == "remote-mqpu") {
              // Get source code
              py::object source_code = get_source_code(inspect, func);

              /*
              // Combine the function source and its call
              std::ostringstream code;
              code << R"(
# Define the objective function
def objective_function(x):
  return (x - 3) ** 2 + 4

class Optimizer:
  def __init__(self, func):
    self.func = func
  def optimize(self, initial_guess):
    current_guess = initial_guess
    learning_rate = 0.1
    for _ in range(100):  # Run 100 iterations
      gradient = 2 * (current_guess - 3)
      current_guess -= learning_rate * gradient
    return current_guess

optimizer = Optimizer(objective_function)

result, parameter = optimizer.optimize(0), 1.234
              )";
              */
              
              // std::string combined_code = code.str();
              std::string combined_code = source_code.cast<std::string>() + func_call;

              std::cout << combined_code << std::endl;

              // Get the parent frame info
              py::object parent_frame_info = get_parent_frame_info(inspect);

              // Get locals and globals for the current frame
              auto [locals, globals] =
                  get_locals_and_globals(parent_frame_info);

              // Use for compiling the source code
              // py::object builtins = py::module::import("builtins");
              // std::string compiled_code =
              // py::str(get_compiled_code(combined_code,
              // builtins)).cast<std::string>();

              std::cout << "Executing python code" << std::endl;
              std::cout << "Locals:\n" << locals << std::endl;
              std::cout << "Globals:\n" << globals << std::endl;
              // py::module traceback = py::module::import("traceback");
              // try {
              //   py::exec(combined_code, globals, locals);
              // } catch (py::error_already_set &e) {
              //   std::cerr << "Caught an error: " << e.what() << std::endl;
              //   if (e.trace()) {
              //     std::cerr << "Traceback:" << std::endl;
              //     std::cerr << e.trace() << std::endl;
              //   }
              //   py::list tb_list = traceback.attr("format_tb")(e.trace());
              //   std::vector<std::string> tb_strings;
              //   for (py::handle str : tb_list) {
              //     tb_strings.push_back(str.cast<std::string>());
              //   }
              //   std::cout << "Traceback:\n" << std::endl;
              //   for (const auto &tb_str : tb_strings) {
              //     std::cout << tb_str;
              //   }
              //   e.restore();
              //   throw;
              //   // throw std::runtime_error("Failed to execute code: " +
              //   //                          std::string(e.what()));
              // }
              // std::cout << "Finished executing python code" << std::endl;
              // std::cout << "Locals:\n" << locals << std::endl;
              // std::cout << "Globals:\n" << globals << std::endl;

              // Serialize the compiled code, locals, and globals
              SerializedCodeExecutionContext *serialized_code_execution_object =
                  get_serialized_code(combined_code, locals, globals);

              // Call the REST API to /job
              call_rest_api(serialized_code_execution_object, locals, globals);

              // return a empty tuple to suppress the compile error for now
              // TODO Need to return the proper value from response
              return std::make_tuple<double, std::vector<double>>(
                  0.0, std::vector<double>());
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
