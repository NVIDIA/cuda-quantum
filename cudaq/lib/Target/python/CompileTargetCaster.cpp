/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompileTargetCaster.h"
#include "Reflection.h"
#include <cctype>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <utility>

namespace {

namespace nb = nanobind;
using cudaq::reflection::typeName;

/// The Python type called `name` in `cudaq.core._target`.
nb::object pyType(std::string_view name) {
  static auto *module =
      new nb::module_(nb::module_::import_("cudaq.core._target"));
  return module->attr(nb::str(name.data(), name.size()));
}

/// The Python name of the C++ field `name`: `eraseQEC` becomes `erase_qec`.
std::string pythonName(std::string_view name) {
  std::string snake;
  for (std::size_t i = 0; i < name.size(); ++i) {
    bool wordStart = i > 0 && std::isupper(name[i]) &&
                     (std::islower(name[i - 1]) ||
                      (i + 1 < name.size() && std::islower(name[i + 1])));
    if (wordStart)
      snake += '_';
    snake += std::tolower(name[i]);
  }
  return snake;
}

/// Runs `f`, which builds a Python object, and reports an exception as a
/// pending Python error and a null handle, as nanobind expects of `from_cpp`.
template <typename F>
nb::handle releaseOrNull(F &&f) noexcept {
  try {
    return f().release();
  } catch (nb::python_error &e) {
    e.restore();
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  }
  return nb::handle();
}

/// The nanobind name of `T`.
template <typename T>
constexpr auto typeDescr = []<std::size_t... I>(std::index_sequence<I...>) {
  return nb::detail::descr<sizeof...(I)>(typeName<T>[I]...);
}(std::make_index_sequence<typeName<T>.size()>());

// `NB_TYPE_CASTER` refers to names of both namespaces unqualified.
using namespace nanobind;
using namespace nanobind::detail;

/// Converts the struct `T` to and from the dataclass of the same name.
template <typename T>
struct struct_caster {
  NB_TYPE_CASTER(T, typeDescr<T>)

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      if (!isinstance(src, pyType(typeName<T>)))
        return false;

      // Ensure dataclass invariants are satisfied.
      if (hasattr(src, "verify"))
        src.attr("verify")();

      bool ok = true;
      cudaq::reflection::forEachField(value, [&](std::string_view name,
                                                 auto &field) {
        // Short-circuit if an earlier field failed to convert.
        if (!ok)
          return;

        using Field = std::remove_reference_t<decltype(field)>;
        make_caster<Field> caster{};
        ok = caster.from_python(getattr(src, pythonName(name).c_str(), none()),
                                flags, cleanup);
        if (!ok || !caster.template can_cast<Field>()) {
          ok = false;
          return;
        }

        field = caster.operator cast_t<Field>();
      });
      return ok;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(const T &value, rv_policy, cleanup_list *) noexcept {
    return releaseOrNull([&] {
      dict kwargs;
      cudaq::reflection::forEachField(
          value, [&](std::string_view name, const auto &field) {
            kwargs[pythonName(name).c_str()] = cast(field);
          });
      return pyType(typeName<T>)(**kwargs);
    });
  }
};

/// Converts the enum `E` to and from the Python enum of the same name. Accepts
/// a member of the enum or its name.
template <typename E>
struct enum_caster {
  NB_TYPE_CASTER(E, typeDescr<E>)

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    try {
      std::string name;
      if (!try_cast(src, name)) {
        if (!isinstance(src, pyType(typeName<E>)))
          return false;
        name = cast<std::string>(src.attr("name"));
      }
      auto enumerator = cudaq::reflection::enumFromName<E>(name);
      if (enumerator)
        value = *enumerator;
      return enumerator.has_value();
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(E value, rv_policy, cleanup_list *) noexcept {
    return releaseOrNull([&]() -> object {
      return pyType(typeName<E>)
          .attr(std::string(cudaq::reflection::enumName(value)).c_str());
    });
  }
};
} // namespace

namespace nanobind::detail {

///-----------------------------------------------------------------------------
/// Nanobind casters based on compile-time reflection
///-----------------------------------------------------------------------------

/// Register a nanobind caster for `T` using `struct_caster` or `enum_caster`.
#define CUDAQ_REFLECTED_CASTER(kind, T)                                        \
  template <>                                                                  \
  struct type_caster<T> : kind##_caster<T> {};

CUDAQ_REFLECTED_CASTER(struct, cudaq::config::UncheckedProfile)
CUDAQ_REFLECTED_CASTER(struct, cudaq::config::QuakeReferenceSemanticsProfile)
CUDAQ_REFLECTED_CASTER(struct, cudaq::config::QuakeUnmappedProfile)
CUDAQ_REFLECTED_CASTER(struct, cudaq::config::QuakeMappedProfile)
CUDAQ_REFLECTED_CASTER(struct, cudaq::config::CustomPipelineLowering)
CUDAQ_REFLECTED_CASTER(struct, cudaq::config::CompilationStage)
CUDAQ_REFLECTED_CASTER(enum, cudaq::NoiseSupport)
CUDAQ_REFLECTED_CASTER(enum, cudaq::config::CostModel)
CUDAQ_REFLECTED_CASTER(enum, cudaq::config::LoopUnrolling)

#undef CUDAQ_REFLECTED_CASTER

///-----------------------------------------------------------------------------
/// Custom casters (e.g. support casting from multiple native Python types)
///-----------------------------------------------------------------------------

/// `MeasurementForm` is a `str` in Python.
template <>
struct type_caster<cudaq::config::MeasurementForm>
    : enum_caster<cudaq::config::MeasurementForm> {
  static constexpr auto Name = const_name("str");

  static handle from_cpp(cudaq::config::MeasurementForm value, rv_policy,
                         cleanup_list *) noexcept {
    auto name = cudaq::reflection::enumName(value);
    return PyUnicode_FromStringAndSize(name.data(), name.size());
  }
};

/// `InferredLowering` is `None` in Python.
template <>
struct type_caster<cudaq::config::InferredLowering>
    : none_caster<cudaq::config::InferredLowering> {};

/// Also accepts the emission format as a string.
template <>
struct type_caster<cudaq::config::Codegen>
    : struct_caster<cudaq::config::Codegen> {
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (std::string format; try_cast(src, format)) {
      value = {std::move(format), ""};
      return true;
    }
    return struct_caster::from_python(src, flags, cleanup);
  }
};

/// `U3Synthesizer` is a Python callable `(theta, phi, lam, epsilon) -> [str]`.
template <>
struct type_caster<cudaq::config::U3Synthesizer> {
  NB_TYPE_CASTER(cudaq::config::U3Synthesizer,
                 const_name("Callable[[float, float, float, float], "
                            "Sequence[str]]"))

  /// Calls the Python callable and reports a Python exception as an error.
  struct PyU3Synthesizer {
    /// Copies of a `CompileTarget` share the callable without the GIL. The
    /// platform keeps its target in a static, which outlives the interpreter:
    /// then there is no reference count left to decrement.
    std::shared_ptr<PyObject> callable;

    explicit PyU3Synthesizer(handle src)
        : callable(src.inc_ref().ptr(), [](PyObject *obj) {
            if (!Py_IsInitialized())
              return;
            gil_scoped_acquire gil;
            Py_DECREF(obj);
          }) {}

    cudaq::config::U3SynthesisResult
    operator()(double theta, double phi, double lambda, double epsilon) const {
      gil_scoped_acquire gil;
      try {
        return {cast<std::vector<std::string>>(
                    handle(callable.get())(theta, phi, lambda, epsilon)),
                std::nullopt};
      } catch (const std::exception &e) {
        return {{}, e.what()};
      }
    }
  };

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!PyCallable_Check(src.ptr()))
      return false;
    value.fn = std::make_shared<const cudaq::config::U3Synthesizer::Fn>(
        PyU3Synthesizer(src));
    return true;
  }

  static handle from_cpp(const cudaq::config::U3Synthesizer &value, rv_policy,
                         cleanup_list *) noexcept {
    if (const auto *py =
            value.fn ? value.fn->target<PyU3Synthesizer>() : nullptr)
      return handle(py->callable.get()).inc_ref();
    return none().release();
  }
};

///-----------------------------------------------------------------------------
/// Implement `CompileTarget` caster (declared in header file)
///-----------------------------------------------------------------------------

bool type_caster<cudaq::CompileTarget>::from_python(
    handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
  struct_caster<cudaq::CompileTarget> caster;
  if (!caster.from_python(src, flags, cleanup))
    return false;
  value = std::move(caster.value);
  return true;
}

handle
type_caster<cudaq::CompileTarget>::from_cpp(const cudaq::CompileTarget &value,
                                            rv_policy policy,
                                            cleanup_list *cleanup) noexcept {
  return struct_caster<cudaq::CompileTarget>::from_cpp(value, policy, cleanup);
}

} // namespace nanobind::detail
