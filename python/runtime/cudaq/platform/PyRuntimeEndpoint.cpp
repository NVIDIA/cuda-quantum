/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PyRuntimeEndpoint.h"
#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>
#include <nanobind/stl/string.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cudaq;

/// Get the class nameof the Python object 'obj'
static std::string getPythonClassName(const nanobind::object &obj) {
  try {
    auto classObj = obj.attr("__class__");
    if (nanobind::hasattr(classObj, "__name__"))
      return nanobind::cast<std::string>(classObj.attr("__name__"));
  } catch (...) {
  }
  return "";
}

/// Wrap our Python object in a shared pointer so that it can be copied around
/// without GIL.
using PyObj = std::shared_ptr<nanobind::object>;
static PyObj makeEndpointHandle(nanobind::object obj) {

  // Our object handle can outlive the interpreter, in which case there is no
  // refcount left to decrement.
  auto objDestructor = +[](nanobind::object *obj) {
    if (!obj || !obj->is_valid()) {
      delete obj;
      return;
    }

    if (!Py_IsInitialized()) {
      (void)obj->release();
      delete obj;
      return;
    }

    nanobind::gil_scoped_acquire gil;
    delete obj;
  };
  return std::shared_ptr<nanobind::object>(new nanobind::object(std::move(obj)),
                                           objDestructor);
}

namespace {

/// The `cudaq::KernelArgs` view handed to a Python endpoint.
///
/// Unlike `KernelArgs`, this does not support packed arguments (a C++-only
/// path). It carries the argument types recovered from the compiled module so
/// that scalar arguments can be converted back to Python values.
struct PyKernelArgs {
  std::span<void *const> args;
  std::optional<std::vector<mlir::Type>> argTypes;
  mlir::ModuleOp module;

  PyKernelArgs() = default;

  PyKernelArgs(KernelArgs kargs, const CompiledModule &compiled) {
    if (kargs.hasPacked())
      CUDAQ_WARN("Packed arguments are not supported for Python runtime "
                 "endpoints. Ignoring arguments.");
    if (auto typeErasedArgs = kargs.getTypeErased())
      args = typeErasedArgs.value();
    parseSignature(compiled);
  }

  /// Convert a single kernel argument to a Python object.
  static nanobind::object convertArg(mlir::ModuleOp module, mlir::Type ty,
                                     void *data) {
    if (!data)
      return nanobind::none();
    if (!mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType>(ty))
      throw std::runtime_error("unsupported argument type: " +
                               mlirTypeToString(ty));
    return convertResult(module, ty, reinterpret_cast<char *>(data));
  }

  std::string repr() const {
    if (!argTypes)
      return "KernelArgs(<unknown args>)";

    std::string out = "KernelArgs([";
    for (std::size_t i = 0; i < argTypes->size(); ++i) {
      if (i)
        out += ", ";
      mlir::Type ty = (*argTypes)[i];
      void *data = i < args.size() ? args[i] : nullptr;
      try {
        nanobind::object obj = convertArg(module, ty, data);
        out += nanobind::cast<std::string>(
            nanobind::steal<nanobind::object>(PyObject_Repr(obj.ptr())));
      } catch (const std::exception &) {
        out += "<instance of " + mlirTypeToString(ty) + ">";
      }
    }
    out += "])";
    return out;
  }

private:
  /// Recover the kernel argument types from the compiled module's MLIR.
  void parseSignature(const CompiledModule &compiled) {
    auto mlirArt = compiled.getMlir();
    if (!mlirArt)
      return;
    mlir::ModuleOp modOp =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    if (!modOp)
      return;
    auto funcOp = getKernelFuncOp</*noThrow=*/true>(modOp, compiled.getName());
    if (!funcOp)
      return;
    auto inputs = funcOp.getFunctionType().getInputs();
    module = modOp;
    argTypes = std::vector<mlir::Type>(inputs.begin(), inputs.end());
  }
};

/// Map launch policies onto Python endpoint protocols.
template <launch_policy Policy>
struct PyProtocol {};

template <>
struct PyProtocol<sample_policy> {
  static constexpr const char *Method = "sample";

  static nanobind::dict kwargs(const sample_policy &policy) {
    nanobind::dict kw;
    kw["shots_count"] = policy.options.shots;
    kw["explicit_measurements"] = policy.options.explicit_measurements;
    return kw;
  }
};

template <>
struct PyProtocol<observe_policy> {
  static constexpr const char *Method = "observe";

  static nanobind::dict kwargs(const observe_policy &policy) {
    nanobind::dict kw;
    if (policy.options.shots >= 0)
      kw["shots_count"] = policy.options.shots;
    kw["spin_operator"] =
        nanobind::cast(policy.spin, nanobind::rv_policy::copy);
    return kw;
  }
};

/// Subset of launch policies that are supported in Python.
template <typename Policy>
concept PyLaunchPolicy = requires { PyProtocol<Policy>::Method; };

/// Dispatch a launch of \p policy to the Python object held in \p impl.
template <PyLaunchPolicy Policy>
typename Policy::result_type pyLaunch(std::any &impl, const Policy &policy,
                                      const CompiledModule &module,
                                      KernelArgs args) {
  using Protocol = PyProtocol<Policy>;

  nanobind::gil_scoped_acquire gil;

  auto &obj = *std::any_cast<PyObj &>(impl);
  if (!nanobind::hasattr(obj, Protocol::Method)) {
    // We expect the method to be available but it isn't - the user probably
    // mutated the endpoint after setting it.
    auto objClassName = getPythonClassName(obj);
    throw std::runtime_error(
        std::string("Expected runtime endpoint") +
        (objClassName.empty() ? "" : " of type '" + objClassName + "'") +
        " to implement '" + Protocol::Method +
        "'. Was the runtime endpoint mutated? To fix this error, call "
        "`cudaq.set_runtime_endpoint` with the new object.");
  }

  CUDAQ_INFO("Dispatching a '{}' launch to the Python runtime endpoint.",
             get_policy_name(policy));

  // Hand ownership of (a copy of) the module to Python
  auto pyModule =
      nanobind::cast(CompiledModule(module), nanobind::rv_policy::move);
  auto pyArgs = nanobind::cast(PyKernelArgs(std::move(args), module),
                               nanobind::rv_policy::move);

  auto kwargs = Protocol::kwargs(policy);
  auto result = obj.attr(Protocol::Method)(pyModule, pyArgs, **kwargs);

  return nanobind::cast<typename Policy::result_type>(result);
}

} // namespace

static RuntimeEndpoint makeRuntimeEndpoint(nanobind::object obj) {
  nanobind::gil_scoped_acquire gil;
  RuntimeEndpoint endpoint;
  endpoint.dispatch = detail::DispatchTable<all_policies>::create(
      // Note: this fixes the set of supported policies at construction time.
      // This means we currently don't support changing the set of supported
      // policies after `set_runtime_endpoint` is called.
      [&obj]<typename Policy>() -> detail::launch_fn_type<Policy> {
        if constexpr (PyLaunchPolicy<Policy>) {
          if (!nanobind::hasattr(obj, PyProtocol<Policy>::Method))
            return nullptr;
          return &pyLaunch<Policy>;
        } else {
          return nullptr;
        }
      });
  endpoint.isSimulator = nanobind::cast<bool>(obj.attr("is_simulator"));
  endpoint.isRemote = nanobind::cast<bool>(obj.attr("is_remote"));
  endpoint.isEmulated = nanobind::cast<bool>(obj.attr("is_emulated"));
  endpoint.impl = makeEndpointHandle(std::move(obj));
  return endpoint;
}

void cudaq::bindRuntimeEndpoint(nanobind::module_ &mod) {
  nanobind::class_<PyKernelArgs>(mod, "KernelArgs",
                                 "The processed arguments of a kernel launch.")
      .def(
          "__len__",
          [](const PyKernelArgs &self) -> std::size_t {
            if (!self.argTypes)
              throw std::runtime_error("unknown number of arguments");
            return self.argTypes->size();
          },
          "The number of kernel arguments.")
      .def(
          "__getitem__",
          [](const PyKernelArgs &self, std::size_t i) -> nanobind::object {
            if (!self.argTypes)
              throw std::runtime_error("unknown argument type");
            if (i >= self.argTypes->size() || i >= self.args.size())
              throw nanobind::index_error();
            return PyKernelArgs::convertArg(self.module, (*self.argTypes)[i],
                                            self.args[i]);
          },
          "Convert the argument at the given index to a Python value. "
          "Currently only supports a limited set of types.")
      .def("__repr__", &PyKernelArgs::repr);

  mod.def(
      "set_runtime_endpoint",
      [](nanobind::object endpoint, std::size_t qpu_id) {
        get_platform().setRuntimeEndpoint(
            makeRuntimeEndpoint(std::move(endpoint)), qpu_id);
      },
      nanobind::arg("endpoint"), nanobind::arg("qpu_id") = 0,
      "Route kernel launches on the given QPU to a Python object implementing "
      "one or more of the cudaq._experimental runtime endpoint protocols.");
}
