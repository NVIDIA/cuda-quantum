//===- DialectFabric.cpp - Fabric dialect Python bindings --------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Nanobind bindings for the Fabric dialect.  Populate-only; driven from
// _qlx_ext.cpp's NB_MODULE like DialectQLX.cpp.  Uses the public
// PyConcreteType / PyConcreteAttribute CRTP pattern from
// mlir/Bindings/Python/IRCore.h so that concrete dialect types/attributes
// are real nanobind subclasses of mlir.ir.Type / mlir.ir.Attribute.
//
//===----------------------------------------------------------------------===//

#include "PopulateSubmodules.h"

#include "qlx-c/Dialect/Fabric.h"

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;

namespace {

inline MlirStringRef toRef(const std::string &s) {
  return MlirStringRef{s.data(), s.size()};
}

inline std::string fromRef(MlirStringRef ref) {
  return std::string(ref.data, ref.length);
}

//===---------------------------------------------------------------------===//
// Code-symbol-parameterized types: fabric.patch<@code>, fabric.syndrome<@code>
//===---------------------------------------------------------------------===//

template <typename DerivedTy, MlirType (*GetFn)(MlirContext, MlirStringRef)>
struct PyFabricCodeParamType : public PyConcreteType<DerivedTy> {
  using Base = PyConcreteType<DerivedTy>;
  using ClassTy = typename Base::ClassTy;
  using PyConcreteType<DerivedTy>::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &codeSymbol, MlirContext ctx) {
          return GetFn(ctx, toRef(codeSymbol));
        },
        nb::arg("code_symbol"), nb::arg("context").none() = nb::none());
  }
};

struct PyPatchType
    : public PyFabricCodeParamType<PyPatchType, fabricPatchTypeGet> {
  static constexpr IsAFunctionTy isaFunction = fabricTypeIsAPatch;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricPatchTypeGetTypeID;
  static constexpr const char *pyClassName = "PatchType";
  using PyFabricCodeParamType::PyFabricCodeParamType;
};

struct PySyndromeType
    : public PyFabricCodeParamType<PySyndromeType, fabricSyndromeTypeGet> {
  static constexpr IsAFunctionTy isaFunction = fabricTypeIsASyndrome;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricSyndromeTypeGetTypeID;
  static constexpr const char *pyClassName = "SyndromeType";
  using PyFabricCodeParamType::PyFabricCodeParamType;
};

//===---------------------------------------------------------------------===//
// fabric.resource<T|CCZ|CS>
//===---------------------------------------------------------------------===//

struct PyResourceStateType : public PyConcreteType<PyResourceStateType> {
  static constexpr IsAFunctionTy isaFunction = fabricTypeIsAResourceState;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricResourceStateTypeGetTypeID;
  static constexpr const char *pyClassName = "ResourceStateType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &resource, MlirContext ctx) {
          MlirType t = fabricResourceStateTypeGet(ctx, toRef(resource));
          if (mlirTypeIsNull(t))
            throw nb::value_error(("Invalid resource: " + resource).c_str());
          return t;
        },
        nb::arg("resource"), nb::arg("context").none() = nb::none());
  }
};

//===---------------------------------------------------------------------===//
// Nullary types
//===---------------------------------------------------------------------===//

template <typename DerivedTy, MlirType (*GetFn)(MlirContext)>
struct PyFabricNullaryType : public PyConcreteType<DerivedTy> {
  using Base = PyConcreteType<DerivedTy>;
  using ClassTy = typename Base::ClassTy;
  using PyConcreteType<DerivedTy>::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get", [](MlirContext ctx) { return GetFn(ctx); },
        nb::arg("context").none() = nb::none());
  }
};

#define FABRIC_NULLARY_TYPE(CLS, GETTER, IS_A, GET_TYPEID, PY_NAME)            \
  struct CLS : public PyFabricNullaryType<CLS, GETTER> {                       \
    static constexpr IsAFunctionTy isaFunction = IS_A;                         \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction = GET_TYPEID;       \
    static constexpr const char *pyClassName = PY_NAME;                        \
    using PyFabricNullaryType::PyFabricNullaryType;                            \
  }

FABRIC_NULLARY_TYPE(PyBitType, fabricBitTypeGet, fabricTypeIsABit,
                    fabricBitTypeGetTypeID, "BitType");
FABRIC_NULLARY_TYPE(PySlotType, fabricSlotTypeGet, fabricTypeIsASlot,
                    fabricSlotTypeGetTypeID, "SlotType");
FABRIC_NULLARY_TYPE(PyMachineType, fabricMachineTypeGet, fabricTypeIsAMachine,
                    fabricMachineTypeGetTypeID, "MachineType");

#undef FABRIC_NULLARY_TYPE

//===---------------------------------------------------------------------===//
// Enum attributes
//===---------------------------------------------------------------------===//

template <typename DerivedTy,
          MlirAttribute (*GetFn)(MlirContext, MlirStringRef),
          MlirStringRef (*ValueFn)(MlirAttribute)>
struct PyFabricEnumAttr : public PyConcreteAttribute<DerivedTy> {
  using Base = PyConcreteAttribute<DerivedTy>;
  using ClassTy = typename Base::ClassTy;
  using PyConcreteAttribute<DerivedTy>::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &value, MlirContext ctx) {
          MlirAttribute attr = GetFn(ctx, toRef(value));
          if (mlirAttributeIsNull(attr))
            throw nb::value_error((std::string("Invalid ") +
                                   DerivedTy::pyClassName + ": " + value)
                                      .c_str());
          return attr;
        },
        nb::arg("value"), nb::arg("context").none() = nb::none());
    c.def_prop_ro("value", [](MlirAttribute self) -> std::string {
      return fromRef(ValueFn(self));
    });
  }
};

#define FABRIC_ENUM_ATTR(CLS, GETTER, IS_A, VALUE_GET, GET_TYPEID, PY_NAME)    \
  struct CLS : public PyFabricEnumAttr<CLS, GETTER, VALUE_GET> {               \
    static constexpr IsAFunctionTy isaFunction = IS_A;                         \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction = GET_TYPEID;       \
    static constexpr const char *pyClassName = PY_NAME;                        \
    using PyFabricEnumAttr::PyFabricEnumAttr;                                  \
  }

// Dialect-prefixed pyClassName values keep nanobind's per-domain type
// registry collision-free with QLX (which also has RoleAttr).  Friendly
// aliases (RoleAttr, ...) are added to the submodule below.
FABRIC_ENUM_ATTR(PyFabricPartitionAttr, fabricPartitionAttrGet,
                 fabricAttributeIsAPartition, fabricPartitionAttrGetValue,
                 fabricPartitionAttrGetTypeID, "FabricPartitionAttr");
FABRIC_ENUM_ATTR(PyFabricRoleAttr, fabricRoleAttrGet, fabricAttributeIsARole,
                 fabricRoleAttrGetValue, fabricRoleAttrGetTypeID,
                 "FabricRoleAttr");
FABRIC_ENUM_ATTR(PyFabricPrepAttr, fabricPrepAttrGet, fabricAttributeIsAPrep,
                 fabricPrepAttrGetValue, fabricPrepAttrGetTypeID,
                 "FabricPrepAttr");
FABRIC_ENUM_ATTR(PyFabricMergeBasisAttr, fabricMergeBasisAttrGet,
                 fabricAttributeIsAMergeBasis, fabricMergeBasisAttrGetValue,
                 fabricMergeBasisAttrGetTypeID, "FabricMergeBasisAttr");
FABRIC_ENUM_ATTR(PyFabricBoundaryAttr, fabricBoundaryAttrGet,
                 fabricAttributeIsABoundary, fabricBoundaryAttrGetValue,
                 fabricBoundaryAttrGetTypeID, "FabricBoundaryAttr");
FABRIC_ENUM_ATTR(PyFabricResourceAttr, fabricResourceAttrGet,
                 fabricAttributeIsAResource, fabricResourceAttrGetValue,
                 fabricResourceAttrGetTypeID, "FabricResourceAttr");

#undef FABRIC_ENUM_ATTR

//===---------------------------------------------------------------------===//
// Composite attributes
//===---------------------------------------------------------------------===//

struct PyFabricFloorplanAttr
    : public PyConcreteAttribute<PyFabricFloorplanAttr> {
  static constexpr IsAFunctionTy isaFunction = fabricAttributeIsAFloorplan;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricFloorplanAttrGetTypeID;
  static constexpr const char *pyClassName = "FabricFloorplanAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &layout, const std::vector<int64_t> &params,
           MlirContext ctx) {
          MlirAttribute attr = fabricFloorplanAttrGet(
              ctx, toRef(layout), static_cast<intptr_t>(params.size()),
              params.data());
          if (mlirAttributeIsNull(attr))
            throw nb::value_error(("Invalid layout: " + layout).c_str());
          return attr;
        },
        nb::arg("layout"), nb::arg("params"),
        nb::arg("context").none() = nb::none());
  }
};

struct PyFabricFlowAttr : public PyConcreteAttribute<PyFabricFlowAttr> {
  static constexpr IsAFunctionTy isaFunction = fabricAttributeIsAFlow;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricFlowAttrGetTypeID;
  static constexpr const char *pyClassName = "FabricFlowAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &x_to, const std::string &z_to, MlirContext ctx) {
          return fabricFlowAttrGet(ctx, toRef(x_to), toRef(z_to));
        },
        nb::arg("x_to"), nb::arg("z_to"),
        nb::arg("context").none() = nb::none());
  }
};

struct PyFabricSpecOnlyAttr : public PyConcreteAttribute<PyFabricSpecOnlyAttr> {
  static constexpr IsAFunctionTy isaFunction = fabricAttributeIsASpecOnly;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      fabricSpecOnlyAttrGetTypeID;
  static constexpr const char *pyClassName = "FabricSpecOnlyAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &name, MlirContext ctx) {
          return fabricSpecOnlyAttrGet(ctx, toRef(name));
        },
        nb::arg("name"), nb::arg("context").none() = nb::none());
    c.def_prop_ro("name", [](MlirAttribute self) -> std::string {
      return fromRef(fabricSpecOnlyAttrGetName(self));
    });
  }
};

} // namespace

void qlx::python::populateFabricSubmodule(nb::module_ &m) {
  // Types
  PyPatchType::bind(m);
  PySyndromeType::bind(m);
  PyResourceStateType::bind(m);
  PyBitType::bind(m);
  PySlotType::bind(m);
  PyMachineType::bind(m);

  // Enum attributes (registered with dialect-prefixed names; aliased to
  // their friendly Fabric-side spelling below).
  PyFabricPartitionAttr::bind(m);
  PyFabricRoleAttr::bind(m);
  PyFabricPrepAttr::bind(m);
  PyFabricMergeBasisAttr::bind(m);
  PyFabricBoundaryAttr::bind(m);
  PyFabricResourceAttr::bind(m);
  PyFabricFloorplanAttr::bind(m);
  PyFabricFlowAttr::bind(m);
  PyFabricSpecOnlyAttr::bind(m);

  m.attr("PartitionAttr") = m.attr("FabricPartitionAttr");
  m.attr("RoleAttr") = m.attr("FabricRoleAttr");
  m.attr("PrepAttr") = m.attr("FabricPrepAttr");
  m.attr("MergeBasisAttr") = m.attr("FabricMergeBasisAttr");
  m.attr("BoundaryAttr") = m.attr("FabricBoundaryAttr");
  m.attr("ResourceAttr") = m.attr("FabricResourceAttr");
  m.attr("FloorplanAttr") = m.attr("FabricFloorplanAttr");
  m.attr("FlowAttr") = m.attr("FabricFlowAttr");
  m.attr("SpecOnlyAttr") = m.attr("FabricSpecOnlyAttr");
}
