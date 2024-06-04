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
#include <regex>
#include "common/RemoteKernelExecutor.h"

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

std::string get_base64_encoded_str_using_pickle(const std::string &source_code) {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");

  py::bytes serialized_code = pickle.attr("dumps")(source_code);
  py::object encoded_code = base64.attr("b64encode")(serialized_code);
  return encoded_code.cast<std::string>();
}

std::string get_base64_encoded_str(const py::dict &namespace_dict) {
  py::object json = py::module_::import("json");
  
  py::exec(R"(
    import json

    class CustomEncoder(json.JSONEncoder):
      def default(self, obj):
        try:
          return super().default(obj)
        except TypeError:
          return str(obj)
  )");
  py::object custom_encoder = py::globals()["CustomEncoder"];
  py::object json_string = json.attr("dumps")(namespace_dict, py::arg("cls")=custom_encoder);
  std::cout << "JSON string: \n" << json_string << std::endl;
  std::string json_cpp_string = json_string.cast<std::string>();
  std::cout << "JSON CPP string: \n" << json_string << std::endl;
  return llvm::encodeBase64(json_cpp_string);
}

std::string get_base64_encoded_str_using_pickle(const py::dict &namespace_dict) {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");

  py::bytes serialized_namespace = pickle.attr("dumps")(namespace_dict);
  py::object encoded_namespace = base64.attr("b64encode")(serialized_namespace);
  return encoded_namespace.cast<std::string>();
}

json serialize_data(std::string &source_code, py::dict &locals,
                    py::dict &globals) {
  auto g_name = globals.attr("pop")("__name__");
  auto g_loader = globals.attr("pop")("__loader__");
  auto g_annotations = globals.attr("pop")("__annotations__");
  auto g_builtins = globals.attr("pop")("__builtins__");
  auto g_file = globals.attr("pop")("__file__");
  // auto g_kernel = globals.attr("pop")("kernel");
  // auto g_hamiltonian = globals.attr("pop")("hamiltonian");
  auto g_objective_function = globals.attr("pop")("objective_function");
  std::string encoded_code_str = get_base64_encoded_str(source_code);
  std::string encoded_locals_str = get_base64_encoded_str(locals);
  std::string encoded_globals_str = get_base64_encoded_str(globals);

  // globals["kernel"] = g_kernel;
  
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

py::dict get_filtered_dict(const py::dict &namespace_dict) {
  py::dict filtered_dict;

  for (auto item : namespace_dict) {
    py::object key = item.first.cast<py::object>();
    py::object value = item.second.cast<py::object>();

    if (!value.is_none()) {
      filtered_dict[key] = value;
      // std::cout << "\nkey: " << key.cast<std::string>() << ", value: " << value;
    }
  }

  return filtered_dict;
}

std::tuple<py::dict, py::dict>
get_locals_and_globals(const py::object &parent_frame_info) {
  py::dict locals;
  py::dict globals;
  try {
    locals = get_filtered_dict(parent_frame_info.attr("frame").attr("f_locals"));
    globals = get_filtered_dict(parent_frame_info.attr("frame").attr("f_globals"));
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

SerializedCodeExecutionContext get_serialized_code(std::string &source_code, py::dict &locals,
                         py::dict &globals) {
  // Serialize the source code, locals, and globals
  json serialized_data, empty_json;
  try {
    serialized_data = serialize_data(source_code, locals, globals);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to serialized data: " +
                             std::string(e.what()));
  }

  SerializedCodeExecutionContext serializedCodeExecutionContext;
  serializedCodeExecutionContext.source_code = serialized_data["source_code"];
  serializedCodeExecutionContext.locals = serialized_data["locals"];
  serializedCodeExecutionContext.globals = serialized_data["globals"];

  return serializedCodeExecutionContext;
}

py::object extract_function_and_call(const py::function &func) {
    // Import necessary Python modules
    py::module inspect = py::module::import("inspect");
    py::module ast = py::module::import("ast");
    py::module textwrap = py::module::import("textwrap");
    py::module astor = py::module::import("astor");

    // Get the source code of the function
    py::object source_code_obj = inspect.attr("getsource")(func.attr("__global__"));
    std::string source_code = source_code_obj.cast<std::string>();
    
    // Get the calling function's source code
    py::object current_frame = inspect.attr("currentframe")();
    py::list outer_frames = inspect.attr("getouterframes")(current_frame);
    py::tuple calling_frame_info = outer_frames[0].cast<py::tuple>();
    py::object calling_frame = calling_frame_info[0];
    py::list calling_source_info = inspect.attr("getsourcelines")(calling_frame).cast<py::tuple>();
    py::list calling_source_lines = calling_source_info[0].cast<py::list>();
    
    // Combine the source lines into a single string
    std::string calling_source_code;
    for (auto line : calling_source_lines) {
      calling_source_code += line.cast<std::string>();
    }

    // Parse the calling function's source code into an AST
    py::object parsed_ast = ast.attr("parse")(calling_source_code);

    // Helper function to get the function name
    std::string function_name = py::str(func.attr("__name__"));

    py::exec(R"(
def extract_calls(node, call_lines, function_name, calling_source_lines):
  ast_Call = ast.Call
  ast_Name = ast.Name

  if isinstance(node, ast_Call):
    func_node = node.func
    if isinstance(func_node, ast_Name):
      name = func_node.id
      if name == function_name:
        lineno = node.lineno
        call_lines.append(calling_source_lines[lineno - 1])
    )");

    py::object extract_calls_py = py::globals()["extract_calls"];

    // Define a visitor class in Python to traverse the AST
    py::object NodeVisitor = ast.attr("NodeVisitor");

    std::vector<std::string> call_lines;
    
    auto visitor = NodeVisitor();
    visitor.attr("visit")(parsed_ast, py::make_tuple(extract_calls_py, py::cast(call_lines), py::str(func.attr("__name__")), calling_source_lines));
    
    // Combine the source code and the function calls into a new code string
    std::cout << "Converting source to string" << std::endl;
    std::string combined_code;// = py::str(source);
    for (const std::string& line : call_lines) {
        combined_code += "\n" + line;
    }
    std::cout << combined_code << std::endl;
    
    return py::cast(combined_code);
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

void call_rest_api(SerializedCodeExecutionContext *serializeCodeExecutionObject, const py::dict &locals, const py::dict &globals) {
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client 
    = cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest");
  
  std::unordered_map<std::string, std::string> map;
  map.emplace("url", "http://localhost:11030//");

  m_client->setConfig(map);

  py::object kernel = locals["kernel"];
  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  auto &platform = get_platform();
  std::cout << "Platform: " << platform.is_simulator() << std::endl;
  ExecutionContext ctx("sample", 1); // platform.get_exec_ctx();
  // std::cout << "ctx: " << ctx << std::endl;
  using namespace mlir;

  ModuleOp mod = unwrap(kernelMod);
  std::cout << "Mod: " << mod->getContext() << std::endl;
  auto *mlirContext = mod->getContext();

  // auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);

  // py::list attributes = kernel.attr("__dict__").attr("items")();
  // for (auto item : attributes) {
  //   py::tuple item_arr = item.cast<py::tuple>();
  //   auto key = item_arr[0];
  //   auto value = item_arr[1];
  //   std::cout << key.cast<std::string>() << ": " << py::str(value).cast<std::string>() << std::endl;
  // }

  std::cout << "Making a REST request..." << std::endl;
  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(*mlirContext, ctx, *serializeCodeExecutionObject, "custatevec-fp64", kernelName,
                              nullptr, nullptr, 0, &errorMsg);
  
  if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
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
            // cudaq_target == "remote-mqpu"
            if (true) {
              // Get source code
              py::object source_code = get_source_code(inspect, func);

              
              // Combine the function source and its call
//               std::ostringstream code;
//               code << R"(
// # Define the objective function
// def objective_function(x):
//   return (x - 3) ** 2 + 4

// class Optimizer:
//   def __init__(self, func):
//     self.func = func
//   def optimize(self, initial_guess):
//     current_guess = initial_guess
//     learning_rate = 0.1
//     for _ in range(100):  # Run 100 iterations
//       gradient = 2 * (current_guess - 3)
//       current_guess -= learning_rate * gradient
//     return current_guess

// optimizer = Optimizer(objective_function)

// result, parameter = optimizer.optimize(0), 1.234
//               )";
              
              // std::string combined_code = code.str();

              std::ostringstream code;
              code << R"(
def objective_function(parameter_vector: List[float], hamiltonian=hamiltonian, gradient_strategy=gradient, kernel=kernel) -> Tuple[float, List[float]]:
    print('In objective function...')
    # Call `cudaq.observe` on the spin operator and ansatz at the
    # optimizer provided parameters. This will allow us to easily
    # extract the expectation value of the entire system in the
    # z-basis.

    # We define the call to `cudaq.observe` here as a lambda to
    # allow it to be passed into the gradient strategy as a
    # function. If you were using a gradient-free optimizer,
    # you could purely define `cost = cudaq.observe().expectation()`.
    get_result = lambda parameter_vector: cudaq.observe(kernel, hamiltonian, parameter_vector).expectation()
    # `cudaq.observe` returns a `cudaq.ObserveResult` that holds the
    # counts dictionary and the `expectation`.
    cost = get_result(parameter_vector)
    print(f"<H> = {cost}")
    # Compute the gradient vector using `cudaq.gradients.STRATEGY.compute()`.
    gradient_vector = gradient_strategy.compute(parameter_vector, get_result, cost)

    # Return the (cost, gradient_vector) tuple.
    return cost, gradient_vector

cudaq.set_random_seed(13)  # make repeatable
energy, parameter = optimizer.optimize(dimensions=1, function=objective_function)
              )";

              // std::string combined_code = source_code.cast<std::string>() + "\n" + func_call;
              // py::object combined_code = extract_function_and_call(func);
              std::string combined_code = code.str();

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

              // std::cout << "Executing python code" << std::endl;
              std::cout << "Combined code:\n" << combined_code << std::endl;
              std::cout << "Locals:\n" << locals << std::endl;
              // std::cout << "Globals:\n" << globals << std::endl;
              try {
                py::exec(py::str(combined_code), globals);
              } catch (py::error_already_set &e) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

                py::object traceback = py::module_::import("traceback");

                py::object format_exception = traceback.attr("format_exception");
                py::object formatted_list = format_exception(py::reinterpret_steal<py::object>(ptype),
                                                              py::reinterpret_steal<py::object>(pvalue),
                                                              py::reinterpret_steal<py::object>(ptraceback));

                py::object formatted = py::str("").attr("join")(formatted_list);

                std::string traceback_str = formatted.cast<std::string>();
                std::cerr << "Exception occurred: " << e.what() << std::endl;
                std::cerr << "Traceback (most recent call back)" << traceback_str << std::endl;

                PyErr_Clear();
              }
              // Serialize the compiled code, locals, and globals
              // SerializedCodeExecutionContext serialized_code_execution_object = 
              //               get_serialized_code(combined_code, locals, globals);

              // SerializedCodeExecutionContext *serialized_code_execution_object_ptr = &serialized_code_execution_object;

              // Call the REST API to /job
              // call_rest_api(serialized_code_execution_object_ptr, locals, globals);

              // delete(&serialized_code_execution_object);

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
