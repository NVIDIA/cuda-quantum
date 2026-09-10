/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PopulateSubmodules.h"

#include "qlx-c/Dialect/Event.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

namespace {

inline MlirStringRef toRef(const std::string &value) {
  return MlirStringRef{value.data(), value.size()};
}

inline std::string fromRef(MlirStringRef ref) {
  return std::string(ref.data, ref.length);
}

} // namespace

void qlx::python::populateEventSubmodule(nb::module_ &module) {
  mlir_type_subclass(module, "HandleType", eventTypeIsAHandle,
                     eventHandleTypeGetTypeID)
      .def_staticmethod(
          "get",
          [](MlirType payload, const std::string &ownership,
             std::optional<MlirAttribute> stream, MlirContext context) {
            MlirAttribute streamAttribute =
                stream.value_or(MlirAttribute{nullptr});
            MlirType type = eventHandleTypeGet(
                context, payload, toRef(ownership), streamAttribute);
            if (mlirTypeIsNull(type))
              throw nb::value_error(
                  "Invalid !event.handle: `stream`, if present, must be a "
                  "symbol reference attribute");
            return type;
          },
          nb::arg("payload"), nb::arg("ownership"),
          nb::arg("stream").none() = nb::none(),
          nb::arg("context").none() = nb::none())
      .def_property_readonly(
          "payload",
          [](MlirType self) { return eventHandleTypeGetPayload(self); })
      .def_property_readonly("ownership",
                             [](MlirType self) {
                               return fromRef(
                                   eventHandleTypeGetOwnership(self));
                             })
      .def_property_readonly("stream", [](MlirType self) {
        MlirAttribute stream = eventHandleTypeGetStream(self);
        if (mlirAttributeIsNull(stream))
          return std::optional<MlirAttribute>{};
        return std::optional<MlirAttribute>{stream};
      });
}
