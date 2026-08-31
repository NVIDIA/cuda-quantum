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
// Nanobind bindings for the Fabric dialect. Populate-only; driven from
// _qlx_ext.cpp's NB_MODULE like DialectQLX.cpp.
//
// Uses mlir_type_subclass / mlir_attribute_subclass from NanobindAdaptors
// so these are pure-Python subclasses of cudaq.mlir.ir.Type / Attribute
// (same pattern as CUDA-Q's quake/cc bindings). PyConcreteType /
// PyConcreteAttribute cannot be used from a separate extension in the
// shared `cudaq` nanobind domain.
//
//===----------------------------------------------------------------------===//

#include "PopulateSubmodules.h"

#include "qlx-c/Dialect/Fabric.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

namespace {

inline MlirStringRef toRef(const std::string &s) {
  return MlirStringRef{s.data(), s.size()};
}

inline std::string fromRef(MlirStringRef ref) {
  return std::string(ref.data, ref.length);
}

void bindCodeParamType(nb::module_ &m, const char *pyName,
                       bool (*isa)(MlirType),
                       MlirType (*getFn)(MlirContext, MlirStringRef),
                       MlirTypeID (*typeIdFn)()) {
  mlir_type_subclass(m, pyName, isa, typeIdFn)
      .def_staticmethod(
          "get",
          [getFn](const std::string &codeSymbol, MlirContext ctx) {
            return getFn(ctx, toRef(codeSymbol));
          },
          nb::arg("code_symbol"), nb::arg("context").none() = nb::none());
}

void bindNullaryType(nb::module_ &m, const char *pyName, bool (*isa)(MlirType),
                     MlirType (*getFn)(MlirContext),
                     MlirTypeID (*typeIdFn)()) {
  mlir_type_subclass(m, pyName, isa, typeIdFn)
      .def_staticmethod(
          "get", [getFn](MlirContext ctx) { return getFn(ctx); },
          nb::arg("context").none() = nb::none());
}

void bindEnumAttr(nb::module_ &m, const char *pyName, const char *alias,
                  bool (*isa)(MlirAttribute),
                  MlirAttribute (*getFn)(MlirContext, MlirStringRef),
                  MlirStringRef (*valueFn)(MlirAttribute),
                  MlirTypeID (*typeIdFn)()) {
  mlir_attribute_subclass(m, pyName, isa, typeIdFn)
      .def_staticmethod(
          "get",
          [getFn, pyName](const std::string &value, MlirContext ctx) {
            MlirAttribute attr = getFn(ctx, toRef(value));
            if (mlirAttributeIsNull(attr))
              throw nb::value_error(
                  (std::string("Invalid ") + pyName + ": " + value).c_str());
            return attr;
          },
          nb::arg("value"), nb::arg("context").none() = nb::none())
      .def_property_readonly(
          "value", [valueFn](MlirAttribute self) { return fromRef(valueFn(self)); });
  m.attr(alias) = m.attr(pyName);
}

} // namespace

void qlx::python::populateFabricSubmodule(nb::module_ &m) {
  bindCodeParamType(m, "PatchType", fabricTypeIsAPatch, fabricPatchTypeGet,
                    fabricPatchTypeGetTypeID);
  bindCodeParamType(m, "SyndromeType", fabricTypeIsASyndrome,
                    fabricSyndromeTypeGet, fabricSyndromeTypeGetTypeID);

  mlir_type_subclass(m, "ResourceStateType", fabricTypeIsAResourceState,
                     fabricResourceStateTypeGetTypeID)
      .def_staticmethod(
          "get",
          [](const std::string &resource, MlirContext ctx) {
            MlirType t = fabricResourceStateTypeGet(ctx, toRef(resource));
            if (mlirTypeIsNull(t))
              throw nb::value_error(("Invalid resource: " + resource).c_str());
            return t;
          },
          nb::arg("resource"), nb::arg("context").none() = nb::none());

  bindNullaryType(m, "BitType", fabricTypeIsABit, fabricBitTypeGet,
                  fabricBitTypeGetTypeID);
  bindNullaryType(m, "SlotType", fabricTypeIsASlot, fabricSlotTypeGet,
                  fabricSlotTypeGetTypeID);
  bindNullaryType(m, "MachineType", fabricTypeIsAMachine, fabricMachineTypeGet,
                  fabricMachineTypeGetTypeID);

  // Dialect-prefixed class names keep the per-domain registry collision-free
  // with QLX; friendly aliases match the historical Fabric-side spelling.
  bindEnumAttr(m, "FabricPartitionAttr", "PartitionAttr",
               fabricAttributeIsAPartition, fabricPartitionAttrGet,
               fabricPartitionAttrGetValue, fabricPartitionAttrGetTypeID);
  bindEnumAttr(m, "FabricRoleAttr", "RoleAttr", fabricAttributeIsARole,
               fabricRoleAttrGet, fabricRoleAttrGetValue,
               fabricRoleAttrGetTypeID);
  bindEnumAttr(m, "FabricPrepAttr", "PrepAttr", fabricAttributeIsAPrep,
               fabricPrepAttrGet, fabricPrepAttrGetValue,
               fabricPrepAttrGetTypeID);
  bindEnumAttr(m, "FabricMergeBasisAttr", "MergeBasisAttr",
               fabricAttributeIsAMergeBasis, fabricMergeBasisAttrGet,
               fabricMergeBasisAttrGetValue, fabricMergeBasisAttrGetTypeID);
  bindEnumAttr(m, "FabricBoundaryAttr", "BoundaryAttr",
               fabricAttributeIsABoundary, fabricBoundaryAttrGet,
               fabricBoundaryAttrGetValue, fabricBoundaryAttrGetTypeID);
  bindEnumAttr(m, "FabricResourceAttr", "ResourceAttr",
               fabricAttributeIsAResource, fabricResourceAttrGet,
               fabricResourceAttrGetValue, fabricResourceAttrGetTypeID);

  mlir_attribute_subclass(m, "FabricFloorplanAttr",
                          fabricAttributeIsAFloorplan,
                          fabricFloorplanAttrGetTypeID)
      .def_staticmethod(
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
  m.attr("FloorplanAttr") = m.attr("FabricFloorplanAttr");

  mlir_attribute_subclass(m, "FabricFlowAttr", fabricAttributeIsAFlow,
                          fabricFlowAttrGetTypeID)
      .def_staticmethod(
          "get",
          [](const std::string &x_to, const std::string &z_to,
             MlirContext ctx) {
            return fabricFlowAttrGet(ctx, toRef(x_to), toRef(z_to));
          },
          nb::arg("x_to"), nb::arg("z_to"),
          nb::arg("context").none() = nb::none());
  m.attr("FlowAttr") = m.attr("FabricFlowAttr");

  mlir_attribute_subclass(m, "FabricSpecOnlyAttr", fabricAttributeIsASpecOnly,
                          fabricSpecOnlyAttrGetTypeID)
      .def_staticmethod(
          "get",
          [](const std::string &name, MlirContext ctx) {
            return fabricSpecOnlyAttrGet(ctx, toRef(name));
          },
          nb::arg("name"), nb::arg("context").none() = nb::none())
      .def_property_readonly("name", [](MlirAttribute self) -> std::string {
        return fromRef(fabricSpecOnlyAttrGetName(self));
      });
  m.attr("SpecOnlyAttr") = m.attr("FabricSpecOnlyAttr");
}
