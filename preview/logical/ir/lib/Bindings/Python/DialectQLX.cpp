/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PopulateSubmodules.h"

#include "qlx-c/Dialect/QLX.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

// Out-of-tree dialect extensions must not include IRCore.h /
// PyConcreteAttribute. Those CRTP types emit a local PyAttribute typeinfo
// that nanobind treats as distinct from CUDA-Q's _mlir.so (Python loads
// extensions RTLD_LOCAL), so registering a subclass aborts with a
// "critical nanobind error". mlir_attribute_subclass looks up
// cudaq.mlir.ir.Attribute by name instead.

namespace {

inline MlirStringRef toRef(const std::string &value) {
  return MlirStringRef{value.data(), value.size()};
}

inline std::string fromRef(MlirStringRef ref) {
  return std::string(ref.data, ref.length);
}

} // namespace

void qlx::python::populateQLXSubmodule(nb::module_ &module) {
  mlir_attribute_subclass(module, "QLXPauliAttr", qlxAttributeIsAPauli,
                          qlxPauliAttrGetTypeID)
      .def_staticmethod(
          "get",
          [](const std::string &value, MlirContext context) {
            MlirAttribute attr = qlxPauliAttrGet(context, toRef(value));
            if (mlirAttributeIsNull(attr))
              throw nb::value_error(
                  (std::string("Invalid QLXPauliAttr: ") + value).c_str());
            return attr;
          },
          nb::arg("value"), nb::arg("context").none() = nb::none())
      .def_property_readonly("value", [](MlirAttribute self) -> std::string {
        return fromRef(qlxPauliAttrGetValue(self));
      });
  module.attr("PauliAttr") = module.attr("QLXPauliAttr");

  module.def(
      "set_inherent_attr",
      [](MlirOperation operation, const std::string &name,
         MlirAttribute attribute) {
        mlirOperationSetInherentAttributeByName(operation, toRef(name),
                                                attribute);
      },
      nb::arg("op"), nb::arg("name"), nb::arg("attr"),
      "Set an inherent (property-style) attribute on an operation by name.");
}
