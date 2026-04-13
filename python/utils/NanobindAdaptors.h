/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"

#include "llvm/ADT/Twine.h"

namespace nb = nanobind;

// Type casters for MLIR C-API types.
namespace nanobind {
namespace detail {

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute.
static nb::object mlirApiObjectToCapsule(nb::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return nb::borrow<nb::object>(apiObject);
  if (!nb::hasattr(apiObject, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto repr = nb::repr(apiObject);
    throw nb::type_error((llvm::Twine("Expected an MLIR object (got ") +
                          std::string(nb::str(repr).c_str()) + ").")
                             .str()
                             .c_str());
  }
  return apiObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

/// Casts object <-> MlirAffineMap.
template <>
struct type_caster<MlirAffineMap> {
  NB_TYPE_CASTER(MlirAffineMap, const_name("MlirAffineMap"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToAffineMap(capsule.ptr());
      return !mlirAffineMapIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirAffineMap v, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object capsule =
          nb::steal<nb::object>(mlirPythonAffineMapToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("AffineMap")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object <-> MlirAttribute.
template <>
struct type_caster<MlirAttribute> {
  NB_TYPE_CASTER(MlirAttribute, const_name("MlirAttribute"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToAttribute(capsule.ptr());
      return !mlirAttributeIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirAttribute v, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object capsule =
          nb::steal<nb::object>(mlirPythonAttributeToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Attribute")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object -> MlirContext.
template <>
struct type_caster<MlirContext> {
  NB_TYPE_CASTER(MlirContext, const_name("MlirContext"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      if (src.is_none()) {
        src = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("Context")
                  .attr("current");
      }
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToContext(capsule.ptr());
      return !mlirContextIsNull(value);
    } catch (...) {
      return false;
    }
  }
};

/// Casts object <-> MlirDialectRegistry.
template <>
struct type_caster<MlirDialectRegistry> {
  NB_TYPE_CASTER(MlirDialectRegistry, const_name("MlirDialectRegistry"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToDialectRegistry(capsule.ptr());
      return !mlirDialectRegistryIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirDialectRegistry v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object capsule =
          nb::steal<nb::object>(mlirPythonDialectRegistryToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("DialectRegistry")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object <-> MlirLocation.
template <>
struct type_caster<MlirLocation> {
  NB_TYPE_CASTER(MlirLocation, const_name("MlirLocation"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      if (src.is_none()) {
        src = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("Location")
                  .attr("current");
      }
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToLocation(capsule.ptr());
      return !mlirLocationIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirLocation v, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object capsule =
          nb::steal<nb::object>(mlirPythonLocationToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Location")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object <-> MlirModule.
template <>
struct type_caster<MlirModule> {
  NB_TYPE_CASTER(MlirModule, const_name("MlirModule"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToModule(capsule.ptr());
      return !mlirModuleIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirModule v, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object capsule = nb::steal<nb::object>(mlirPythonModuleToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Module")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object <-> MlirOperation.
template <>
struct type_caster<MlirOperation> {
  NB_TYPE_CASTER(MlirOperation, const_name("MlirOperation"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToOperation(capsule.ptr());
      return !mlirOperationIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirOperation v, rv_policy, cleanup_list *) noexcept {
    if (v.ptr == nullptr)
      return nb::none().release();
    try {
      nb::object capsule =
          nb::steal<nb::object>(mlirPythonOperationToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Operation")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object <-> MlirValue.
template <>
struct type_caster<MlirValue> {
  NB_TYPE_CASTER(MlirValue, const_name("MlirValue"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToValue(capsule.ptr());
      return !mlirValueIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirValue v, rv_policy, cleanup_list *) noexcept {
    if (v.ptr == nullptr)
      return nb::none().release();
    try {
      nb::object capsule = nb::steal<nb::object>(mlirPythonValueToCapsule(v));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Value")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

/// Casts object -> MlirPassManager.
template <>
struct type_caster<MlirPassManager> {
  NB_TYPE_CASTER(MlirPassManager, const_name("MlirPassManager"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToPassManager(capsule.ptr());
      return !mlirPassManagerIsNull(value);
    } catch (...) {
      return false;
    }
  }
};

/// Casts object <-> MlirType.
template <>
struct type_caster<MlirType> {
  NB_TYPE_CASTER(MlirType, const_name("MlirType"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      nb::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToType(capsule.ptr());
      return !mlirTypeIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirType t, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object capsule = nb::steal<nb::object>(mlirPythonTypeToCapsule(t));
      return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr("Type")
          .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
          .release();
    } catch (...) {
      return handle();
    }
  }
};

} // namespace detail
} // namespace nanobind

namespace mlir {
namespace python {
namespace nanobind_adaptors {

/// Provides a facility like nb::class_ for defining a new class in a scope,
/// but this allows extension of an arbitrary Python class, defining methods
/// on it in a similar way. These are not "real" nanobind classes but pure
/// Python classes with no relation to a concrete C++ class.
class pure_subclass {
public:
  pure_subclass(nb::handle scope, const char *derivedClassName,
                const nb::object &superClass) {
    nb::object pyType = nb::borrow<nb::object>((PyObject *)&PyType_Type);
    nb::object metaclass = pyType(superClass);
    nb::dict attributes;

    thisClass =
        metaclass(derivedClassName, nb::make_tuple(superClass), attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &...extra) {
    nb::object cf(nb::cpp_function(std::forward<Func>(f), nb::name(name),
                                   nb::arg("self"), extra...));
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &...extra) {
    nb::object cf(nb::cpp_function(std::forward<Func>(f), nb::name(name),
                                   nb::arg("self"), extra...));
    auto builtinProperty = nb::borrow<nb::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &...extra) {
    nb::object cf(
        nb::cpp_function(std::forward<Func>(f), nb::name(name), extra...));
    thisClass.attr(name) = nb::steal(PyStaticMethod_New(cf.ptr()));
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_classmethod(const char *name, Func &&f,
                                 const Extra &...extra) {
    nb::object cf(nb::cpp_function(std::forward<Func>(f), nb::name(name),
                                   nb::scope(thisClass), extra...));
    thisClass.attr(name) = nb::steal<nb::object>(PyClassMethod_New(cf.ptr()));
    return *this;
  }

  nb::object get_class() const { return thisClass; }

protected:
  nb::object superClass;
  nb::object thisClass;
};

/// Creates a custom subclass of mlir.ir.Type, implementing a casting
/// constructor and type checking methods.
class mlir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirType);

  /// Subclasses by looking up the super-class dynamically.
  mlir_type_subclass(nb::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction)
      : mlir_type_subclass(scope, typeClassName, isaFunction,
                           nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                               .attr("Type")) {}

  /// Subclasses with a provided mlir.ir.Type super-class.
  mlir_type_subclass(nb::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction, const nb::object &superCls)
      : pure_subclass(scope, typeClassName, superCls) {
    std::string captureTypeName(typeClassName);
    nb::object newCf(nb::cpp_function(
        [superCls, isaFunction, captureTypeName](nb::object cls,
                                                 nb::object otherType) {
          MlirType rawType = nb::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr = std::string(nb::str(nb::repr(otherType)).c_str());
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          nb::object self = superCls.attr("__new__")(cls, otherType);
          return self;
        },
        nb::name("__new__"), nb::arg("cls"), nb::arg("cast_from_type")));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        nb::arg("other_type"));
  }
};

/// Creates a custom subclass of mlir.ir.Attribute, implementing a casting
/// constructor and type checking methods.
class mlir_attribute_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirAttribute);

  /// Subclasses by looking up the super-class dynamically.
  mlir_attribute_subclass(nb::handle scope, const char *attrClassName,
                          IsAFunctionTy isaFunction)
      : mlir_attribute_subclass(
            scope, attrClassName, isaFunction,
            nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Attribute")) {}

  /// Subclasses with a provided mlir.ir.Attribute super-class.
  mlir_attribute_subclass(nb::handle scope, const char *typeClassName,
                          IsAFunctionTy isaFunction, const nb::object &superCls)
      : pure_subclass(scope, typeClassName, superCls) {
    std::string captureTypeName(typeClassName);
    nb::object newCf(nb::cpp_function(
        [superCls, isaFunction, captureTypeName](nb::object cls,
                                                 nb::object otherAttribute) {
          MlirAttribute rawAttribute = nb::cast<MlirAttribute>(otherAttribute);
          if (!isaFunction(rawAttribute)) {
            auto origRepr =
                std::string(nb::str(nb::repr(otherAttribute)).c_str());
            throw std::invalid_argument(
                (llvm::Twine("Cannot cast attribute to ") + captureTypeName +
                 " (from " + origRepr + ")")
                    .str());
          }
          nb::object self = superCls.attr("__new__")(cls, otherAttribute);
          return self;
        },
        nb::name("__new__"), nb::arg("cls"), nb::arg("cast_from_attr")));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirAttribute other) { return isaFunction(other); },
        nb::arg("other_attribute"));
  }
};

} // namespace nanobind_adaptors
} // namespace python
} // namespace mlir
