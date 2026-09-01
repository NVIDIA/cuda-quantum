/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PopulateSubmodules.h"

#include "qlx-c/Dialect/QLX.h"

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"

#include <nanobind/stl/string.h>

namespace nb = nanobind;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute;

namespace {

inline MlirStringRef toRef(const std::string &value) {
  return MlirStringRef{value.data(), value.size()};
}

inline std::string fromRef(MlirStringRef ref) {
  return std::string(ref.data, ref.length);
}

template <typename DerivedTy,
          MlirAttribute (*GetFn)(MlirContext, MlirStringRef),
          MlirStringRef (*ValueFn)(MlirAttribute)>
struct PyQLXEnumAttr : public PyConcreteAttribute<DerivedTy> {
  using Base = PyConcreteAttribute<DerivedTy>;
  using ClassTy = typename Base::ClassTy;
  using PyConcreteAttribute<DerivedTy>::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &value, MlirContext context) {
          MlirAttribute attr = GetFn(context, toRef(value));
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

struct PyQLXPauliAttr : public PyQLXEnumAttr<PyQLXPauliAttr, qlxPauliAttrGet,
                                             qlxPauliAttrGetValue> {
  static constexpr IsAFunctionTy isaFunction = qlxAttributeIsAPauli;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      qlxPauliAttrGetTypeID;
  static constexpr const char *pyClassName = "QLXPauliAttr";
  using PyQLXEnumAttr::PyQLXEnumAttr;
};

} // namespace

void qlx::python::populateQLXSubmodule(nb::module_ &module) {
  PyQLXPauliAttr::bind(module);
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
