/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "utils/NanobindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include <fmt/core.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;
using namespace mlir;

namespace cudaq {
static bool registered = false;

void registerQuakeDialectAndTypes(nb::module_ &m) {
  auto quakeMod = m.def_submodule("quake");

  quakeMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);

        if (!registered) {
          cudaq::registerCudaqPassesAndPipelines();
          registered = true;
        }
      },
      nb::arg("load") = true, nb::arg("context") = nb::none());

  mlir_type_subclass(
      quakeMod, "RefType",
      [](MlirType type) { return unwrap(type).isa<quake::RefType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext context) {
            return wrap(quake::RefType::get(unwrap(context)));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(
      quakeMod, "MeasureType",
      [](MlirType type) { return unwrap(type).isa<quake::MeasureType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext context) {
            return wrap(quake::MeasureType::get(unwrap(context)));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(
      quakeMod, "MeasurementsType",
      [](MlirType type) { return unwrap(type).isa<quake::MeasurementsType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, std::size_t size, MlirContext context) {
            return wrap(quake::MeasurementsType::get(unwrap(context), size));
          },
          nb::arg("cls"),
          nb::arg("size") = quake::MeasurementsType::kDynamicSize,
          nb::arg("context") = nb::none());

  mlir_type_subclass(
      quakeMod, "VeqType",
      [](MlirType type) { return unwrap(type).isa<quake::VeqType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, std::size_t size, MlirContext context) {
            return wrap(quake::VeqType::get(unwrap(context), size));
          },
          nb::arg("cls"),
          nb::arg("size") = std::numeric_limits<std::size_t>::max(),
          nb::arg("context") = nb::none())
      .def_staticmethod(
          "hasSpecifiedSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.hasSpecifiedSize();
          },
          nb::arg("veqTypeInstance"))
      .def_staticmethod(
          "getSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.getSize();
          },
          nb::arg("veqTypeInstance"));

  quakeMod.def(
      "isConstantQuantumRefType",
      [](MlirType type) {
        return quake::isConstantQuantumRefType(unwrap(type));
      },
      nb::arg("type"));

  quakeMod.def(
      "getAllocationSize",
      [](MlirType type) { return quake::getAllocationSize(unwrap(type)); },
      nb::arg("type"));

  mlir_type_subclass(
      quakeMod, "StruqType",
      [](MlirType type) { return unwrap(type).isa<quake::StruqType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, nb::list aggregateTypes, MlirContext context) {
            SmallVector<Type> inTys;
            for (auto t : aggregateTypes)
              inTys.push_back(unwrap(nb::cast<MlirType>(t)));

            return wrap(quake::StruqType::get(unwrap(context), inTys));
          },
          nb::arg("cls"), nb::arg("aggregateTypes"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "getNamed",
          [](nb::object cls, const std::string &name, nb::list aggregateTypes,
             MlirContext context) {
            SmallVector<Type> inTys;
            for (auto t : aggregateTypes)
              inTys.push_back(unwrap(nb::cast<MlirType>(t)));

            return wrap(quake::StruqType::get(unwrap(context), name, inTys));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("aggregateTypes"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "getTypes",
          [](nb::object cls, MlirType structTy) {
            auto ty = dyn_cast<quake::StruqType>(unwrap(structTy));
            if (!ty)
              throw std::runtime_error(
                  "invalid type passed to StruqType.getTypes(), must be a "
                  "quake.struq");
            std::vector<MlirType> ret;
            for (auto &t : ty.getMembers())
              ret.push_back(wrap(t));
            return ret;
          })
      .def_classmethod("getName", [](nb::object cls, MlirType structTy) {
        auto ty = dyn_cast<quake::StruqType>(unwrap(structTy));
        if (!ty)
          throw std::runtime_error(
              "invalid type passed to StruqType.getName(), must be a "
              "quake.struq");
        return ty.getName().getValue().str();
      });
}

void registerCCDialectAndTypes(nb::module_ &m) {

  auto ccMod = m.def_submodule("cc");

  ccMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle ccHandle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(ccHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ccHandle, context);
        }
      },
      nb::arg("load") = true, nb::arg("context") = nb::none());

  mlir_type_subclass(
      ccMod, "CharspanType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::CharspanType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext context) {
            return wrap(cudaq::cc::CharspanType::get(unwrap(context)));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(
      ccMod, "StateType",
      [](MlirType type) { return unwrap(type).isa<quake::StateType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext context) {
            return wrap(quake::StateType::get(unwrap(context)));
          },
          nb::arg("cls"), nb::arg("context") = nb::none());

  mlir_type_subclass(
      ccMod, "PointerType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::PointerType>(); })
      .def_classmethod(
          "getElementType",
          [](nb::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::PointerType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to PointerType.getElementType(), must "
                  "be cc.ptr type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType, MlirContext context) {
            return wrap(cudaq::cc::PointerType::get(unwrap(context),
                                                    unwrap(elementType)));
          },
          nb::arg("cls"), nb::arg("elementType"),
          nb::arg("context") = nb::none());

  mlir_type_subclass(
      ccMod, "ArrayType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::ArrayType>(); })
      .def_classmethod(
          "getElementType",
          [](nb::object cls, MlirType type) {
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
          [](nb::object cls, MlirType elementType, std::int64_t size,
             MlirContext context) {
            return wrap(cudaq::cc::ArrayType::get(unwrap(context),
                                                  unwrap(elementType), size));
          },
          nb::arg("cls"), nb::arg("elementType"),
          nb::arg("size") = std::numeric_limits<std::int64_t>::min(),
          nb::arg("context") = nb::none());

  mlir_type_subclass(
      ccMod, "StructType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::StructType>(); })
      .def_classmethod(
          "get",
          [](nb::object cls, nb::list aggregateTypes, MlirContext context) {
            SmallVector<Type> inTys;
            for (auto t : aggregateTypes)
              inTys.push_back(unwrap(nb::cast<MlirType>(t)));

            return wrap(cudaq::cc::StructType::get(unwrap(context), inTys));
          },
          nb::arg("cls"), nb::arg("aggregateTypes"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "getNamed",
          [](nb::object cls, const std::string &name, nb::list aggregateTypes,
             MlirContext context) {
            SmallVector<Type> inTys;
            for (auto t : aggregateTypes)
              inTys.push_back(unwrap(nb::cast<MlirType>(t)));

            return wrap(
                cudaq::cc::StructType::get(unwrap(context), name, inTys));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("aggregateTypes"),
          nb::arg("context") = nb::none())
      .def_classmethod(
          "getTypes",
          [](nb::object cls, MlirType structTy) {
            auto ty = dyn_cast<cudaq::cc::StructType>(unwrap(structTy));
            if (!ty)
              throw std::runtime_error(
                  "invalid type passed to StructType.getTypes(), must be a "
                  "cc.struct");
            std::vector<MlirType> ret;
            for (auto &t : ty.getMembers())
              ret.push_back(wrap(t));
            return ret;
          })
      .def_classmethod("getName", [](nb::object cls, MlirType structTy) {
        auto ty = dyn_cast<cudaq::cc::StructType>(unwrap(structTy));
        if (!ty)
          throw std::runtime_error(
              "invalid type passed to StructType.getName(), must be a "
              "cc.struct");
        return ty.getName().getValue().str();
      });

  mlir_type_subclass(
      ccMod, "CallableType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::CallableType>(); })
      .def_classmethod("get",
                       [](nb::object cls, MlirContext context, nb::list inTypes,
                          nb::list resTypes) {
                         // Nanobind builder: make the builder for this type
                         // look like that of a FunctionType.
                         SmallVector<Type> inTys;
                         for (auto t : inTypes)
                           inTys.push_back(unwrap(nb::cast<MlirType>(t)));
                         SmallVector<Type> resTys;
                         for (auto t : resTypes)
                           resTys.push_back(unwrap(nb::cast<MlirType>(t)));

                         auto *ctx = unwrap(context);
                         return wrap(cudaq::cc::CallableType::get(
                             ctx, FunctionType::get(ctx, inTys, resTys)));
                       })
      .def_classmethod("getFunctionType", [](nb::object cls, MlirType type) {
        auto callTy = dyn_cast<cudaq::cc::CallableType>(unwrap(type));
        if (!callTy)
          throw std::runtime_error("must be a cc.callable type!");
        return wrap(callTy.getSignature());
      });

  mlir_type_subclass(
      ccMod, "StdvecType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::StdvecType>(); })
      .def_classmethod(
          "getElementType",
          [](nb::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::StdvecType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to StdvecType.getElementType(), must "
                  "be cc.array type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType, MlirContext context) {
            return wrap(cudaq::cc::StdvecType::get(unwrap(context),
                                                   unwrap(elementType)));
          },
          nb::arg("cls"), nb::arg("elementType"),
          nb::arg("context") = nb::none());
}

void bindRegisterDialects(nb::module_ &mod) {
  registerQuakeDialectAndTypes(mod);
  registerCCDialectAndTypes(mod);

  mod.def("load_intrinsic", [](MlirModule module, std::string name) {
    auto unwrapped = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(unwrapped.getBody());
    if (failed(builder.loadIntrinsic(unwrapped, name)))
      unwrapped.emitError("failed to load intrinsic " + name);
  });

  mod.def("register_all_dialects", [](MlirContext context) {
    DialectRegistry registry;
    registry.insert<quake::QuakeDialect, cudaq::cc::CCDialect>();
    cudaq::opt::registerCodeGenDialect(registry);
    registerAllDialects(registry);
    auto *mlirContext = unwrap(context);
    mlirContext->appendDialectRegistry(registry);
    mlirContext->loadAllAvailableDialects();
  });

  mod.def("gen_vector_of_complex_constant", [](MlirLocation loc,
                                               MlirModule module,
                                               std::string name,
                                               const std::vector<std::complex<
                                                   double>> &values) {
    ModuleOp modOp = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(modOp.getBody());
    SmallVector<std::complex<double>> newValues{values.begin(), values.end()};
    builder.genVectorOfConstants(unwrap(loc), modOp, name, newValues);
  });
}
} // namespace cudaq
