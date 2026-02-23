/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "OpaqueArguments.h"
#include "cudaq/qis/state.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

/// Pack a list of Python typed-wrapper objects into an OpaqueArguments.
/// Each element in \p args must be one of the wrapper types defined in
/// python/cudaq/kernel/kernel_args.py.
inline bool packWrappedArgs(OpaqueArguments &argData, py::list args) {
  for (auto item : args) {
    auto obj = py::reinterpret_borrow<py::object>(item);
    std::string cls =
        obj.attr("__class__").attr("__name__").cast<std::string>();

    if (cls == "Float64Arg") {
      addArgument(argData, obj.attr("value").cast<double>());

    } else if (cls == "Float32Arg") {
      addArgument(argData, obj.attr("value").cast<float>());

    } else if (cls == "IntArg") {
      addArgument(argData, obj.attr("value").cast<std::int64_t>());

    } else if (cls == "BoolArg") {
      addArgument(argData, static_cast<char>(obj.attr("value").cast<bool>()));

    } else if (cls == "ComplexF64Arg") {
      addArgument(argData, obj.attr("value").cast<std::complex<double>>());

    } else if (cls == "ComplexF32Arg") {
      addArgument(argData, obj.attr("value").cast<std::complex<float>>());

    } else if (cls == "PauliWordArg") {
      addArgument(argData, obj.attr("value").cast<pauli_word>().str());

    } else if (cls == "StateArg") {
      auto *stateArg = obj.attr("value").cast<state *>();
      if (stateArg == nullptr)
        throw std::runtime_error("Null cudaq::state* argument passed.");
      auto simState =
          state_helper::getSimulationState(const_cast<state *>(stateArg));
      if (!simState)
        throw std::runtime_error(
            "Error: Unable to retrieve simulation state from cudaq::state. "
            "The state contains no simulation state.");
      if (simState->getKernelInfo().has_value()) {
        state *copyState = new state(*stateArg);
        argData.emplace_back(
            copyState, [](void *ptr) { delete static_cast<state *>(ptr); });
      } else {
        argData.emplace_back(stateArg, [](void *) {});
      }

    } else if (cls == "StructArg") {
      auto mlirTy = obj.attr("mlir_type");
      auto mlirMod = obj.attr("module");
      auto mod = unwrap(mlirMod.cast<MlirModule>());
      auto ty =
          mlir::cast<cudaq::cc::StructType>(unwrap(mlirTy.cast<MlirType>()));
      auto [size, offsets] = getTargetLayout(mod, ty);
      auto memberTys = ty.getMembers();
      auto allocatedArg = std::malloc(size);
      py::object value = obj.attr("value");
      if (ty.getName() == "tuple") {
        auto elements = value.cast<py::tuple>();
        for (std::size_t i = 0; i < offsets.size(); i++)
          handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                     elements[i]);
      } else {
        py::dict attributes = value.attr("__annotations__").cast<py::dict>();
        for (std::size_t i = 0; const auto &[attr_name, unused] : attributes) {
          py::object attr_value =
              value.attr(attr_name.cast<std::string>().c_str());
          handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                     attr_value);
          i++;
        }
      }
      argData.emplace_back(allocatedArg, [](void *ptr) { std::free(ptr); });

    } else if (cls == "VecArg") {
      auto mlirTy = obj.attr("mlir_type");
      auto ty =
          mlir::cast<cudaq::cc::StdvecType>(unwrap(mlirTy.cast<MlirType>()));
      auto list = obj.attr("value").cast<py::list>();
      auto eleTy = ty.getElementType();

      auto appendVectorValue = [&argData]<typename T>(mlir::Type eleTy,
                                                      py::list list) {
        auto allocatedArg = handleVectorElements(eleTy, list);
        argData.emplace_back(allocatedArg, [](void *ptr) {
          delete static_cast<std::vector<T> *>(ptr);
        });
      };

      if (eleTy.isInteger(1)) {
        appendVectorValue.template operator()<char>(eleTy, list);
      } else {
        appendVectorValue.template operator()<std::int64_t>(eleTy, list);
      }

    } else if (cls == "LinkedKernelCapture") {
      auto kernelName = obj.attr("linkedKernel").cast<std::string>();
      kernelName.erase(0, strlen(runtime::cudaqGenPrefixName));
      auto kernelModule = unwrap(obj.attr("qkeModule").cast<MlirModule>());
      OpaqueArguments resolvedArgs;
      argData.emplace_back(
          new runtime::CallableClosureArgument(
              kernelName, kernelModule, std::nullopt, std::move(resolvedArgs)),
          [](void *that) {
            delete static_cast<runtime::CallableClosureArgument *>(that);
          });

    } else if (cls == "DecoratorCapture") {
      py::object decorator = obj.attr("decorator");
      auto kernelName = decorator.attr("uniqName").cast<std::string>();
      auto kernelModule =
          unwrap(decorator.attr("qkeModule").cast<MlirModule>());
      // auto calledFuncOp =
      //     kernelModule.lookupSymbol<mlir::func::FuncOp>(
      //         runtime::cudaqGenPrefixName + kernelName);
      py::list arguments = obj.attr("resolved");
      auto startLiftedArgs = [&]() -> std::optional<unsigned> {
        if (!arguments.empty())
          return decorator.attr("formal_arity")().cast<unsigned>();
        return std::nullopt;
      }();
      OpaqueArguments resolvedArgs;
      if (startLiftedArgs) {
        packWrappedArgs(resolvedArgs, arguments);
      }
      auto *closure = new runtime::CallableClosureArgument(
          kernelName, kernelModule, std::move(startLiftedArgs),
          std::move(resolvedArgs));
      argData.emplace_back(closure, [](void *that) {
        delete static_cast<runtime::CallableClosureArgument *>(that);
      });

    } else {
      throw std::runtime_error(
          "Unknown argument wrapper type: " + cls +
          " for value: " + py::str(obj).cast<std::string>());
    }
  }
  return true;
}

} // namespace cudaq

namespace pybind11 {
namespace detail {

template <>
struct type_caster<cudaq::OpaqueArguments> {
  PYBIND11_TYPE_CASTER(cudaq::OpaqueArguments, const_name("OpaqueArguments"));

  bool load(handle src, bool /*convert*/) {
    if (!isinstance<py::list>(src))
      return false;
    return cudaq::packWrappedArgs(value, reinterpret_borrow<py::list>(src));
  }

  // C++ -> Python direction is not needed.
  static handle cast(cudaq::OpaqueArguments &&, return_value_policy, handle) {
    throw std::runtime_error("OpaqueArguments cannot be converted to Python.");
  }
};

} // namespace detail
} // namespace pybind11
