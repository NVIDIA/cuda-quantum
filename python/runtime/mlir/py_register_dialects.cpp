/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#include <fmt/core.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;
using namespace mlir;

namespace cudaq {
static bool registered = false;

void registerQuakeDialectAndTypes(py::module &m) {
  auto quakeMod = m.def_submodule("quake");

  quakeMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }

        if (!registered) {
          cudaq::opt::registerOptCodeGenPasses();
          cudaq::opt::registerOptTransformsPasses();
          cudaq::opt::registerAggressiveEarlyInlining();
          cudaq::opt::registerUnrollingPipeline();
          cudaq::opt::registerTargetPipelines();
          registered = true;
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(quakeMod, "RefType", [](MlirType type) {
    return isa<quake::RefType>(unwrap(type));
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::RefType::get(unwrap(ctx)));
  });

  mlir_type_subclass(quakeMod, "MeasureType", [](MlirType type) {
    return isa<quake::MeasureType>(unwrap(type));
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::MeasureType::get(unwrap(ctx)));
  });

  mlir_type_subclass(
      quakeMod, "VeqType",
      [](MlirType type) { return isa<quake::VeqType>(unwrap(type)); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, std::size_t size) {
            return wrap(quake::VeqType::get(unwrap(ctx), size));
          },
          py::arg("cls"), py::arg("context"), py::arg("size") = 0)
      .def_staticmethod(
          "hasSpecifiedSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.hasSpecifiedSize();
          },
          py::arg("veqTypeInstance"))
      .def_staticmethod(
          "getSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.getSize();
          },
          py::arg("veqTypeInstance"));
}

void registerCCDialectAndTypes(py::module &m) {

  auto ccMod = m.def_submodule("cc");

  ccMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle ccHandle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(ccHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ccHandle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(ccMod, "CharspanType", [](MlirType type) {
    return isa<cudaq::cc::CharspanType>(unwrap(type));
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(cudaq::cc::CharspanType::get(unwrap(ctx)));
  });

  mlir_type_subclass(
      ccMod, "PointerType",
      [](MlirType type) { return isa<cudaq::cc::PointerType>(unwrap(type)); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::PointerType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to PointerType.getElementType(), must "
                  "be cc.ptr type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::PointerType::get(unwrap(ctx), unwrap(elementType)));
          });

  mlir_type_subclass(
      ccMod, "ArrayType",
      [](MlirType type) { return isa<cudaq::cc::ArrayType>(unwrap(type)); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::ArrayType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to ArrayType.getElementType(), must "
                  "be cc.array type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, MlirType elementType,
             std::int64_t size) {
            return wrap(cudaq::cc::ArrayType::get(unwrap(ctx),
                                                  unwrap(elementType), size));
          },
          py::arg("cls"), py::arg("ctx"), py::arg("elementType"),
          py::arg("size") = std::numeric_limits<std::int64_t>::min());

  mlir_type_subclass(
      ccMod, "StructType",
      [](MlirType type) { return isa<cudaq::cc::StructType>(unwrap(type)); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, py::list aggregateTypes) {
            SmallVector<Type> inTys;
            for (auto &t : aggregateTypes)
              inTys.push_back(unwrap(cast<MlirType>(t)));

            return wrap(cudaq::cc::StructType::get(unwrap(ctx), inTys));
          })
      .def_classmethod("getTypes", [](py::object cls, MlirType structTy) {
        auto ty = dyn_cast<cudaq::cc::StructType>(unwrap(structTy));
        if (!ty)
          throw std::runtime_error(
              "invalid type passed to StructType.getTypes(), must be a "
              "cc.struct");
        std::vector<MlirType> ret;
        for (auto &t : ty.getMembers())
          ret.push_back(wrap(t));
        return ret;
      });

  mlir_type_subclass(
      ccMod, "CallableType",
      [](MlirType type) { return isa<cudaq::cc::CallableType>(unwrap(type)); })
      .def_classmethod("get",
                       [](py::object cls, MlirContext ctx, py::list inTypes) {
                         SmallVector<Type> inTys;
                         for (auto &t : inTypes)
                           inTys.push_back(unwrap(cast<MlirType>(t)));

                         return wrap(cudaq::cc::CallableType::get(
                             unwrap(ctx), FunctionType::get(unwrap(ctx), inTys,
                                                            TypeRange{})));
                       })
      .def_classmethod("getFunctionType", [](py::object cls, MlirType type) {
        return wrap(
            dyn_cast<cudaq::cc::CallableType>(unwrap(type)).getSignature());
      });

  mlir_type_subclass(
      ccMod, "StdvecType",
      [](MlirType type) { return isa<cudaq::cc::StdvecType>(unwrap(type)); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::StdvecType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to StdvecType.getElementType(), must "
                  "be cc.array type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::StdvecType::get(unwrap(ctx), unwrap(elementType)));
          });
}

void bindRegisterDialects(py::module &mod) {
  registerQuakeDialectAndTypes(mod);
  registerCCDialectAndTypes(mod);

  mod.def("load_intrinsic", [](MlirModule module, std::string name) {
    auto unwrapped = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(unwrapped.getBody());
    if (failed(builder.loadIntrinsic(unwrapped, name)))
      unwrapped.emitError("failed to load intrinsic " + name);
  });
}
} // namespace cudaq
