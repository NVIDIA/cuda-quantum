/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_optimizer.h"

#include "common/JsonConvert.h"
#include "common/RemoteKernelExecutor.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq/algorithms/gradients/central_difference.h"
#include "cudaq/algorithms/gradients/forward_difference.h"
#include "cudaq/algorithms/gradients/parameter_shift.h"
#include "cudaq/algorithms/optimizers/ensmallen/ensmallen.h"
#include "cudaq/algorithms/optimizers/nlopt/nlopt.h"
#include "cudaq/optimizers.h"
#include "cudaq/spin_op.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "llvm/Support/Base64.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <regex>

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

std::string
get_base64_encoded_str_using_pickle(const std::string &source_code) {
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
          print('Received an error when encoding the following object...')
          print(obj)
          return str(obj)
  )");
  py::object custom_encoder = py::globals()["CustomEncoder"];
  py::object json_string =
      json.attr("dumps")(namespace_dict, py::arg("cls") = custom_encoder);
  std::string json_cpp_string = json_string.cast<std::string>();
  return llvm::encodeBase64(json_cpp_string);
}

std::string get_base64_encoded_str_using_pickle() {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");

  std::cout << "Serializing the dictionary" << std::endl;
  py::dict serialized_dict;
  for (const auto item : py::globals()) {
    try {
      auto key = item.first;
      auto value = item.second;
      std::cout << key << ":" << value << std::endl;

      py::bytes serialized_value = pickle.attr("dumps")(value);
      serialized_dict[key] = serialized_value;
    } catch (const py::error_already_set &e) {
      std::cout << "Failed to pickle key: " + std::string(e.what())
                << std::endl;
    }
  }

  py::bytes serialized_code = pickle.attr("dumps")(serialized_dict);
  py::object encoded_dict = base64.attr("b64encode")(serialized_code);
  return encoded_dict.cast<std::string>();
}

json serialize_data(std::string &source_code) {
  std::string encoded_code_str = get_base64_encoded_str_using_pickle(source_code);
  std::string encoded_globals_str = get_base64_encoded_str_using_pickle();

  json json_object;
  json_object["source_code"] = encoded_code_str;
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

std::string read_notebook_file(const std::string &file_path) {
  py::dict notebook_content;

  py::object open = py::module_::import("builtins").attr("open");
  py::object json = py::module_::import("json");

  py::object file = open(file_path, "r");
  notebook_content = json.attr("load")(file);

  py::list cells = notebook_content["cells"].cast<py::list>();
  py::list new_cells;

  for (auto &cell : cells) {
    py::dict cell_dict = cell.cast<py::dict>();
    if(cell_dict["cell_type"].cast<std::string>() == "code") {
      py::list source = cell_dict["source"].cast<py::list>();
      py::list new_source;
      for (auto &line : source) {
        std::string line_str = line.cast<std::string>();
        std::size_t comment_pos = line_str.find('#');
        if (comment_pos != std::string::npos) {
          line_str = line_str.substr(0, comment_pos);
        }
        if (!line_str.empty()) {
          new_source.append(line_str + "\n");
        }
      }
      cell_dict["source"] = new_source;
      new_cells.append(cell_dict);
    }
  }

  notebook_content["cells"] = new_cells;

  return py::str(notebook_content);
}

std::string read_python_file(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  std::stringstream buffer;
  std::string line;

  while (std::getline(file, line)) {
    std::size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    buffer << line << '\n';
  }

  return buffer.str();
}

std::string read_file(const std::string &file_path, const bool &is_python_notebook) {
  if (is_python_notebook) {
    return read_notebook_file(file_path);
  } else {
    return read_python_file(file_path);
  }
}

std::tuple<std::string, bool> is_python_notebook(const std::string &substring_to_check) {
  py::dict globals = py::globals();

  for (auto item : py::globals()) {
    std::string key = py::str(item.first);
    if (key.find(substring_to_check) != std::string::npos) {
      return std::make_tuple(key, true);
    }
  }

  return std::make_tuple("", false);
}

std::string get_file_content() {
  // Get the source file of the function
  std::string source_file_path;
  std::string file_content;
  try {
    auto [key, is_notebook] = is_python_notebook("_ipynb_file__");
    if (is_notebook) {
      // TODO Need to pass the key to py::globals()
      source_file_path = py::globals()["__vsc_ipynb_file__"].cast<std::string>();
      file_content = read_file(source_file_path, true);
    } else {
      source_file_path = py::globals()["__file__"].cast<std::string>();
      // Read the entire source file
      file_content = read_file(source_file_path, false);
    }
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source file: " +
                             std::string(e.what()));
  }

  return file_content;
}

std::string get_source_code(const py::function &func) {
  // Get the source code
  py::object inspect = py::module::import("inspect");
  py::object source_code;
  try {
    source_code = inspect.attr("getsource")(func);
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to get source code: " +
                             std::string(e.what()));
  }

  return source_code.cast<std::string>();
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

void print_globals_dict() {
  for (auto item : py::globals()) {
    std::cout << item.first << " : " << item.second << std::endl;
  }
}

void get_filtered_dict() {
  py::dict globals = py::globals();
  std::vector<py::handle> keys_to_remove;

  for (auto it = globals.begin(); it != globals.end(); ++it) {
    std::string key = py::str(it->first);
    std::string key_type = py::repr(py::type::of(it->first));
    std::string value = py::str(it->second);

    if ((key.find("__") != std::string::npos && key != "__builtins__") ||
        it->second.is_none() ||
        (value.find("<module") != std::string::npos && key != "__builtins__") ||
        value.find("typing") != std::string::npos) {
      keys_to_remove.push_back(it->first);
    }
  }

  for (auto &key : keys_to_remove) {
    globals.attr("pop")(key);
  }
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

SerializedCodeExecutionContext get_serialized_code(std::string &source_code) {
  // Serialize the source code, locals, and globals
  json serialized_data;
  SerializedCodeExecutionContext serializedCodeExecutionContext;
  try {
    serialized_data = serialize_data(source_code);
    serializedCodeExecutionContext.source_code = serialized_data["source_code"];
    serializedCodeExecutionContext.globals = serialized_data["globals"];
  } catch (py::error_already_set &e) {
    throw std::runtime_error("Failed to serialized data: " +
                             std::string(e.what()));
  }

  return serializedCodeExecutionContext;
}

py::object extract_function(const py::function &func) {
  // Import necessary Python modules
  py::module inspect = py::module::import("inspect");

  // Get the source code of the function
  py::object source_code_obj = inspect.attr("getsource")(func);

  return source_code_obj;
}

std::vector<std::string> get_kernel_functions(const std::string &code) {
  std::regex kernel_regex(
      R"(@cudaq\.kernel\s+def\s+\w+\([^)]*\):\s*(?:[^#\n]*\n)*?(?:\n\s*\n|$))");
  std::smatch match;
  std::string code_copy = code;
  std::vector<std::string> functions;

  while (std::regex_search(code_copy, match, kernel_regex)) {
    functions.push_back(match.str(0));
    code_copy = match.suffix().str();
  }

  return functions;
}

std::string get_cudaq_target(const std::string &file_content) {
  std::regex cudaq_target_regex(R"(cudaq\.set_target\(\"([^"]+)\".*)");
  std::smatch match;
  std::string cudaq_target;

  if (std::regex_search(file_content, match, cudaq_target_regex) &&
      match.size() > 1) {
    cudaq_target = match.str(1);
  } else {
    throw std::runtime_error("Target not found");
  }

  return cudaq_target;
}

std::string get_imports() {
  py::module sys = py::module::import("sys");
  py::dict sys_modules = sys.attr("modules");
  py::dict globals = py::globals();
  std::string imports_str;

  for (auto item : globals) {
    if (py::isinstance<py::module>(item.second)) {
      py::module mod = item.second.cast<py::module>();
      std::string alias = py::str(item.first);
      std::string name = py::str(mod.attr("__name__"));
      std::cout << "Alias: " << alias << " -> Module: " << name << std::endl;
      if (alias == name)
        imports_str += "import " + name + "\n";
      else
        imports_str += "import " + name + " as " + alias + "\n";
    }
  }
  return imports_str;
}

std::string get_kernels(const std::string &file_content) {
  std::string kernels;

  std::vector<std::string> kernel_functions =
      get_kernel_functions(file_content);
  for (const auto &kernel : kernel_functions) {
    kernels += kernel + "\n";
  }

  return kernels;
}

std::string get_objective_function_call(const std::string &file_content) {
  std::smatch match;
  std::regex objective_function_call_regex(
      R"(.*=\s*optimizer\.optimize\s*\([^)]*function\s*=\s*objective_function\s*\))");

  std::string objective_function_call;
  if (std::regex_search(file_content, match, objective_function_call_regex)) {
    objective_function_call = match.str(0);
  } else {
    throw std::runtime_error("Objective function call not found");
  }

  return objective_function_call;
}

std::string get_required_raw_source_code(const int dim, const py::function &func) {
  // Get file content
  // std::string file_content = get_file_content();

  // Get imports
  std::string imports = get_imports();

  // Get kernels
  // std::string kernels = get_kernels(file_content);

  // Get objective_function call
  // std::string function_call = get_objective_function_call(file_content);

  // Get source code
  std::string source_code = get_source_code(func);

  // Form the Python call to optimizer.optimize
  std::ostringstream os;
  auto obj_func_name = func.attr("__name__").cast<std::string>();
  os << "energy, params_at_energy = optimizer.optimize(" << dim << ", " << obj_func_name << ")\n";
  auto function_call = os.str();

  // Return the combined code
  return imports + "\n" + source_code + "\n" + function_call;
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

// cudaq::optimization_result call_rest_api(
//     SerializedCodeExecutionContext *serializeCodeExecutionObject) {
//   std::unique_ptr<cudaq::RemoteRuntimeClient> m_client =
//       cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest");

//   std::cout << "In call_rest_api" << std::endl;
//   std::unordered_map<std::string, std::string> map;
//   map.emplace("url", "http://localhost:19030//");

//   m_client->setConfig(map);

//   py::object kernel = py::globals()["kernel"];
//   auto kernelName = kernel.attr("name").cast<std::string>();
//   auto kernelMod = kernel.attr("module").cast<MlirModule>();
//   auto &platform = get_platform();
//   std::cout << "Platform: " << platform.is_simulator() << std::endl;
//   ExecutionContext ctx("sample", 1); // platform.get_exec_ctx();
//   // std::cout << "ctx: " << ctx << std::endl;
//   using namespace mlir;

//   ModuleOp mod = unwrap(kernelMod);
//   std::cout << "Mod: " << mod->getContext() << std::endl;
//   auto *mlirContext = mod->getContext();

//   // auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);

//   // py::list attributes = kernel.attr("__dict__").attr("items")();
//   // for (auto item : attributes) {
//   //   py::tuple item_arr = item.cast<py::tuple>();
//   //   auto key = item_arr[0];
//   //   auto value = item_arr[1];
//   //   std::cout << key.cast<std::string>() << ": " <<
//   //   py::str(value).cast<std::string>() << std::endl;
//   // }

//   std::cout << "Making a REST request..." << std::endl;
//   std::string errorMsg;
//   const bool requestOkay = m_client->sendRequest(
//       *mlirContext, ctx, *serializeCodeExecutionObject, "custatevec-fp64",
//       kernelName, nullptr, nullptr, 0, &errorMsg);

//   if (!requestOkay)
//     throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
  
//   return ctx.optResult.value_or(cudaq::optimization_result{});
// }

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
          // objective_function
          // use C++ function
          [](OptimizerT &opt, const int dim, py::function &func) {
            // write a function to detect if cudaq.set_target has been used and
            // fetch its value

            auto &platform = cudaq::get_platform();
            if (platform.supports_remote_serialized_code()) {
              auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", 0);
              platform.set_exec_ctx(ctx.get());

              print_globals_dict();

              std::string combined_code =
                  get_required_raw_source_code(dim, func);

              std::cout << "Combined code\n" << combined_code << std::endl;
              SerializedCodeExecutionContext serialized_code_execution_object =
                  get_serialized_code(combined_code);

              platform.launchSerializedCodeExecution(
                  func.attr("__name__").cast<std::string>(),
                  serialized_code_execution_object);

              platform.reset_exec_ctx();
              auto result = std::move(
                  ctx->optResult.value_or(cudaq::optimization_result{}));
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
