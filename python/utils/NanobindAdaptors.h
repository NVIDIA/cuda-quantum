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

// Type casters for MLIR C-API types.
namespace nanobind {
namespace detail {

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute.
static nanobind::object mlirApiObjectToCapsule(nanobind::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return nanobind::borrow<nanobind::object>(apiObject);
  if (!nanobind::hasattr(apiObject, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto repr = nanobind::repr(apiObject);
    throw nanobind::type_error((llvm::Twine("Expected an MLIR object (got ") +
                                std::string(nanobind::str(repr).c_str()) + ").")
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToAffineMap(capsule.ptr());
      return !mlirAffineMapIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirAffineMap v, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonAffineMapToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToAttribute(capsule.ptr());
      return !mlirAttributeIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirAttribute v, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonAttributeToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
        src = nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("Context")
                  .attr("current");
      }
      nanobind::object capsule = mlirApiObjectToCapsule(src);
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToDialectRegistry(capsule.ptr());
      return !mlirDialectRegistryIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirDialectRegistry v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nanobind::object capsule = nanobind::steal<nanobind::object>(
          mlirPythonDialectRegistryToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
        src = nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("Location")
                  .attr("current");
      }
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToLocation(capsule.ptr());
      return !mlirLocationIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirLocation v, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonLocationToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToModule(capsule.ptr());
      return !mlirModuleIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirModule v, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonModuleToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToOperation(capsule.ptr());
      return !mlirOperationIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirOperation v, rv_policy, cleanup_list *) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none().release();
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonOperationToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToValue(capsule.ptr());
      return !mlirValueIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirValue v, rv_policy, cleanup_list *) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none().release();
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonValueToCapsule(v));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
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
      nanobind::object capsule = mlirApiObjectToCapsule(src);
      value = mlirPythonCapsuleToType(capsule.ptr());
      return !mlirTypeIsNull(value);
    } catch (...) {
      return false;
    }
  }
  static handle from_cpp(MlirType t, rv_policy, cleanup_list *) noexcept {
    try {
      nanobind::object capsule =
          nanobind::steal<nanobind::object>(mlirPythonTypeToCapsule(t));
      return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
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

/// Provides a facility like nanobind::class_ for defining a new class in a
/// scope, but this allows extension of an arbitrary Python class, defining
/// methods on it in a similar way. These are not "real" nanobind classes but
/// pure Python classes with no relation to a concrete C++ class.
class pure_subclass {
public:
  pure_subclass(nanobind::handle scope, const char *derivedClassName,
                const nanobind::object &superClass) {
    nanobind::object pyType =
        nanobind::borrow<nanobind::object>((PyObject *)&PyType_Type);
    nanobind::object metaclass = pyType(superClass);
    nanobind::dict attributes;

    thisClass = metaclass(derivedClassName, nanobind::make_tuple(superClass),
                          attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &...extra) {
    nanobind::object cf(
        nanobind::cpp_function(std::forward<Func>(f), nanobind::name(name),
                               nanobind::arg("self"), extra...));
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &...extra) {
    nanobind::object cf(
        nanobind::cpp_function(std::forward<Func>(f), nanobind::name(name),
                               nanobind::arg("self"), extra...));
    auto builtinProperty =
        nanobind::borrow<nanobind::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &...extra) {
    nanobind::object cf(nanobind::cpp_function(std::forward<Func>(f),
                                               nanobind::name(name), extra...));
    thisClass.attr(name) = nanobind::steal(PyStaticMethod_New(cf.ptr()));
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_classmethod(const char *name, Func &&f,
                                 const Extra &...extra) {
    nanobind::object cf(
        nanobind::cpp_function(std::forward<Func>(f), nanobind::name(name),
                               nanobind::scope(thisClass), extra...));
    thisClass.attr(name) =
        nanobind::steal<nanobind::object>(PyClassMethod_New(cf.ptr()));
    return *this;
  }

  nanobind::object get_class() const { return thisClass; }

protected:
  nanobind::object superClass;
  nanobind::object thisClass;
};

/// Creates a custom subclass of mlir.ir.Type, implementing a casting
/// constructor and type checking methods.
class mlir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirType);

  /// Subclasses by looking up the super-class dynamically.
  mlir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction)
      : mlir_type_subclass(
            scope, typeClassName, isaFunction,
            nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Type")) {}

  /// Subclasses with a provided mlir.ir.Type super-class.
  mlir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     const nanobind::object &superCls)
      : pure_subclass(scope, typeClassName, superCls) {
    std::string captureTypeName(typeClassName);
    nanobind::object newCf(nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](nanobind::object cls,
                                                 nanobind::object otherType) {
          MlirType rawType = nanobind::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr =
                std::string(nanobind::str(nanobind::repr(otherType)).c_str());
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherType);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_type")));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        nanobind::arg("other_type"));
  }
};

/// Creates a custom subclass of mlir.ir.Attribute, implementing a casting
/// constructor and type checking methods.
class mlir_attribute_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirAttribute);

  /// Subclasses by looking up the super-class dynamically.
  mlir_attribute_subclass(nanobind::handle scope, const char *attrClassName,
                          IsAFunctionTy isaFunction)
      : mlir_attribute_subclass(
            scope, attrClassName, isaFunction,
            nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Attribute")) {}

  /// Subclasses with a provided mlir.ir.Attribute super-class.
  mlir_attribute_subclass(nanobind::handle scope, const char *typeClassName,
                          IsAFunctionTy isaFunction,
                          const nanobind::object &superCls)
      : pure_subclass(scope, typeClassName, superCls) {
    std::string captureTypeName(typeClassName);
    nanobind::object newCf(nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](
            nanobind::object cls, nanobind::object otherAttribute) {
          MlirAttribute rawAttribute =
              nanobind::cast<MlirAttribute>(otherAttribute);
          if (!isaFunction(rawAttribute)) {
            auto origRepr = std::string(
                nanobind::str(nanobind::repr(otherAttribute)).c_str());
            throw std::invalid_argument(
                (llvm::Twine("Cannot cast attribute to ") + captureTypeName +
                 " (from " + origRepr + ")")
                    .str());
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherAttribute);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_attr")));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirAttribute other) { return isaFunction(other); },
        nanobind::arg("other_attribute"));
  }
};

} // namespace nanobind_adaptors
} // namespace python
} // namespace mlir
