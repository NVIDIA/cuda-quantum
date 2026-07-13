/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PyDynamicQPU.h"
#include "common/KernelArgs.h"
#include "common/Registry.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "py_alt_launch_kernel.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <functional>
#include <nanobind/stl/string.h>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

// Type alias for `kwargs` argument to pass to nanobind Python function calls.
using KwArgs = nanobind::detail::kwargs_proxy;

// Method names in the Python API. These must match the method names in
// `python/cudaq/experimental/qpu.py`
static const char *const SAMPLE_LAUNCH_NAME = "launch_sample";
static const char *const OBSERVE_LAUNCH_NAME = "launch_observe";
static const char *const SAMPLE_COMPILE_TARGET_NAME =
    "get_compile_target_sample";
static const char *const OBSERVE_COMPILE_TARGET_NAME =
    "get_compile_target_observe";

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PyDynamicQPU, python_dynamic)

namespace {
/// The bindings for the cudaq::KernelArgs type exposed to Python.
///
/// Unlike KernelArgs, PyKernelArgs does not support packed arguments (used in
/// C++). It also stores the types of each argument (parsed from the MLIR
/// module) to allow casting back to Python types
struct PyKernelArgs {
  std::span<void *const> args;
  std::optional<std::vector<mlir::Type>> argTypes;
  mlir::ModuleOp module;

  PyKernelArgs() = default;

  PyKernelArgs(cudaq::KernelArgs kargs, const cudaq::CompiledModule &compiled) {
    if (kargs.hasPacked())
      CUDAQ_WARN("Packed arguments are not supported for Python QPUs. Ignoring "
                 "arguments");
    if (auto typeErasedArgs = kargs.getTypeErased())
      args = typeErasedArgs.value();
    parseSignature(compiled);
  }

private:
  /// Recover the kernel argument types from the compiled module's MLIR.
  void parseSignature(const cudaq::CompiledModule &compiled) {
    auto mlirArt = compiled.getMlir();
    if (!mlirArt)
      return;
    mlir::ModuleOp modOp =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    if (!modOp)
      return;
    auto funcOp =
        cudaq::getKernelFuncOp</*noThrow=*/true>(modOp, compiled.getName());
    if (!funcOp)
      return;
    auto inputs = funcOp.getFunctionType().getInputs();
    module = modOp;
    argTypes = std::vector<mlir::Type>(inputs.begin(), inputs.end());
  }
};
} // namespace

/// Convert a single kernel argument to a Python object.
static nanobind::object convertArg(mlir::ModuleOp module, mlir::Type ty,
                                   void *data) {
  if (!data)
    return nanobind::none();
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType>(ty))
    throw std::runtime_error("unsupported argument type: " +
                             cudaq::mlirTypeToString(ty));
  return cudaq::convertResult(module, ty, reinterpret_cast<char *>(data));
}

template <typename Ret, typename... Args>
static std::optional<std::function<Ret(Args...)>>
getCallable(nanobind::handle obj, const char *name) {
  // Ensure Python and object are alive
  if (!Py_IsInitialized() || !obj.is_valid())
    return std::nullopt;
  // Caller must hold the GIL
  assert(PyGILState_Check());

  if (!nanobind::hasattr(obj, name))
    return std::nullopt;
  std::function<Ret(Args...)> func;
  if (!nanobind::try_cast(obj.attr(name), func)) {
    return std::nullopt;
  }
  return func;
}

cudaq::PyDynamicQPU::~PyDynamicQPU() {
  if (!pyObject.is_valid())
    return;

  // May get cleaned up after Python exit
  if (!Py_IsInitialized()) {
    (void)pyObject.release();
    return;
  }

  nanobind::gil_scoped_acquire gil;
  pyObject = nanobind::object();
}

cudaq::PyDynamicQPU
cudaq::PyDynamicQPU::fromPythonObject(nanobind::object obj) noexcept {
  assert(obj.is_valid());

  PyDynamicQPU qpu;
  qpu.pyObject = obj;

  return qpu;
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::PyDynamicQPU::getCompileTarget(const sample_policy &policy) {
  nanobind::gil_scoped_acquire gil;
  auto callable =
      getCallable<cudaq::CompileTarget>(pyObject, SAMPLE_COMPILE_TARGET_NAME);
  if (!callable)
    throw std::runtime_error(
        "QPU does not implement the SupportsSampleQPU protocol");
  return std::make_unique<cudaq::CompileTarget>((*callable)());
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::PyDynamicQPU::getCompileTarget(const observe_policy &policy) {
  nanobind::gil_scoped_acquire gil;
  auto callable =
      getCallable<cudaq::CompileTarget>(pyObject, OBSERVE_COMPILE_TARGET_NAME);
  if (!callable)
    throw std::runtime_error(
        "QPU does not implement the SupportsObserveQPU protocol");
  return std::make_unique<cudaq::CompileTarget>((*callable)());
}

cudaq::sample_result
cudaq::PyDynamicQPU::launchKernel(const sample_policy &policy,
                                  const CompiledModule &module,
                                  KernelArgs args) {
  nanobind::gil_scoped_acquire gil;
  auto callable =
      getCallable<cudaq::sample_result, const CompiledModule &, PyKernelArgs,
                  KwArgs>(pyObject, SAMPLE_LAUNCH_NAME);
  if (!callable)
    throw std::runtime_error(
        "QPU does not implement the SupportsSampleQPU protocol");
  CUDAQ_INFO("PyDynamicQPU::launchKernel {}", policy.name);
  auto pyArgs = PyKernelArgs(std::move(args), module);
  auto kwargs = nanobind::dict();
  kwargs["shots_count"] = policy.options.shots;
  return (*callable)(module, std::move(pyArgs), **kwargs);
}

cudaq::observe_result
cudaq::PyDynamicQPU::launchKernel(const observe_policy &policy,
                                  const CompiledModule &module,
                                  KernelArgs args) {
  nanobind::gil_scoped_acquire gil;
  auto callable =
      getCallable<cudaq::observe_result, const CompiledModule &, PyKernelArgs,
                  KwArgs>(pyObject, OBSERVE_LAUNCH_NAME);
  if (!callable)
    throw std::runtime_error(
        "QPU does not implement the SupportsObserveQPU protocol");
  CUDAQ_INFO("PyDynamicQPU::launchKernel {}", policy.name);
  auto pyArgs = PyKernelArgs(std::move(args), module);
  auto kwargs = nanobind::dict();
  if (policy.options.shots != -1)
    kwargs["shots_count"] = policy.options.shots;
  return (*callable)(module, std::move(pyArgs), **kwargs);
}

static std::string reprStr(const std::string &s) { return "'" + s + "'"; }

static std::string reprKernelArgs(const PyKernelArgs &self) {
  if (!self.argTypes)
    return "KernelArgs(<unknown args>)";

  std::string out = "KernelArgs([";
  for (std::size_t i = 0; i < self.argTypes->size(); ++i) {
    if (i)
      out += ", ";
    mlir::Type ty = (*self.argTypes)[i];
    void *data = i < self.args.size() ? self.args[i] : nullptr;
    try {
      nanobind::object obj = convertArg(self.module, ty, data);
      out += nanobind::cast<std::string>(
          nanobind::steal<nanobind::object>(PyObject_Repr(obj.ptr())));
    } catch (const std::exception &) {
      out += "<instance of " + cudaq::mlirTypeToString(ty) + ">";
    }
  }
  out += "])";
  return out;
}

static std::string
pipelineConfigRepr(const cudaq::CompileTarget::PipelineConfig &pc) {
  std::ostringstream os;
  os << "PipelineConfig(";
  if (!pc.overridePassPipeline.empty())
    os << "override_pass_pipeline=" << reprStr(pc.overridePassPipeline);
  else {
    os << "high_level_pipeline=" << reprStr(pc.highLevelPipeline)
       << ", mid_level_pipeline=" << reprStr(pc.midLevelPipeline)
       << ", low_level_pipeline=" << reprStr(pc.lowLevelPipeline)
       << ", codegen_translation=" << reprStr(pc.codegenTranslation)
       << ", post_code_gen_passes=" << reprStr(pc.postCodeGenPasses);
  }
  os << ")";
  return os.str();
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  std::ostringstream os;
  os << "CompileTarget(pipeline_config="
     << pipelineConfigRepr(ct.pipelineConfig) << ")";
  return os.str();
}

////// From here onwards, pure bindings code //////

static void bindCompileTarget(nanobind::module_ &mod) {
  nanobind::class_<cudaq::CompileTarget::PipelineConfig>(mod, "PipelineConfig")
      .def(nanobind::init<>())
      .def_rw("override_pass_pipeline",
              &cudaq::CompileTarget::PipelineConfig::overridePassPipeline)
      .def_rw("high_level_pipeline",
              &cudaq::CompileTarget::PipelineConfig::highLevelPipeline)
      .def_rw("mid_level_pipeline",
              &cudaq::CompileTarget::PipelineConfig::midLevelPipeline)
      .def_rw("low_level_pipeline",
              &cudaq::CompileTarget::PipelineConfig::lowLevelPipeline)
      .def_rw("codegen_translation",
              &cudaq::CompileTarget::PipelineConfig::codegenTranslation)
      .def_rw("post_code_gen_passes",
              &cudaq::CompileTarget::PipelineConfig::postCodeGenPasses)
      .def_rw("skip_target_lowering_pipeline",
              &cudaq::CompileTarget::PipelineConfig::skipTargetLoweringPipeline)
      .def_rw("disable_qubit_mapping",
              &cudaq::CompileTarget::PipelineConfig::disableQubitMapping)
      .def("__repr__", pipelineConfigRepr);

  nanobind::class_<cudaq::CompileTarget>(mod, "CompileTarget")
      .def(nanobind::init<>())
      .def_static("default_sample",
                  []() -> cudaq::CompileTarget {
                    return *cudaq::getDefaultCompileTarget(
                        cudaq::sample_policy{});
                  })
      .def_static("default_observe",
                  []() -> cudaq::CompileTarget {
                    return *cudaq::getDefaultCompileTarget(
                        cudaq::observe_policy{});
                  })
      .def_rw("pipeline_config", &cudaq::CompileTarget::pipelineConfig)
      .def("__repr__", compileTargetRepr);
}

static void bindKernelArgs(nanobind::module_ &mod) {
  nanobind::class_<PyKernelArgs>(mod, "KernelArgs",
                                 "Processed arguments for a kernel launch.")
      .def(
          "__len__",
          [](const PyKernelArgs &self) -> std::size_t {
            if (!self.argTypes)
              throw std::runtime_error("unknown number of arguments");
            return self.argTypes->size();
          },
          "The number of kernel arguments. Requires a known signature.")
      .def(
          "__getitem__",
          [](const PyKernelArgs &self, std::size_t i) -> nanobind::object {
            if (!self.argTypes)
              throw std::runtime_error("unknown argument type");
            if (i >= self.argTypes->size() || i >= self.args.size())
              throw nanobind::index_error();
            return convertArg(self.module, (*self.argTypes)[i], self.args[i]);
          },
          "Convert the argument at the given index to a Python value. Requires "
          "a known signature and a supported (scalar) argument type.")
      .def("__repr__", &reprKernelArgs);
}

void cudaq::bindQPUHelperTypes(nanobind::module_ &mod) {
  bindCompileTarget(mod);
  bindKernelArgs(mod);
}
