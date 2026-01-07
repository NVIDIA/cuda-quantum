/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py_ext {

/// Extended python complex object.
///
/// Includes `complex`, `numpy.complex64`, `numpy.complex128`.
class Complex : public pybind11::object {
public:
  PYBIND11_OBJECT_CVT(Complex, object, isComplex_, convert_)

  Complex(double real, double imag)
      : object(PyComplex_FromDoubles(real, imag), stolen_t{}) {
    if (!m_ptr) {
      pybind11::pybind11_fail("Could not allocate complex object!");
    }
  }

  // Allow implicit conversion from complex<double>/complex<float>:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Complex(std::complex<double> value)
      : Complex((double)value.real(), (double)value.imag()) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  Complex(std::complex<float> value)
      : Complex((double)value.real(), (double)value.imag()) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::complex<double>() {
    auto value = PyComplex_AsCComplex(m_ptr);
    return std::complex<double>(value.real, value.imag);
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::complex<float>() {
    auto value = PyComplex_AsCComplex(m_ptr);
    return std::complex<float>(value.real, value.imag);
  }

  static bool isComplex_(PyObject *o) {
    if (PyComplex_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.complex64" || name == "numpy.complex128") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isComplex_(o)) {
      double real = PyComplex_RealAsDouble(o);
      double imag = PyComplex_ImagAsDouble(o);
      ret = PyComplex_FromDoubles(real, imag);
    } else {
      pybind11::set_error(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

/// Extended python float object.
///
/// Includes `float`, `numpy.float64`, `numpy.float32`.
class Float : public pybind11::object {
public:
  PYBIND11_OBJECT_CVT(Float, object, isFloat_, convert_)

  // Allow implicit conversion from float/double:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Float(float value) : object(PyFloat_FromDouble((double)value), stolen_t{}) {
    if (!m_ptr) {
      pybind11::pybind11_fail("Could not allocate float object!");
    }
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  Float(double value = .0)
      : object(PyFloat_FromDouble((double)value), stolen_t{}) {
    if (!m_ptr) {
      pybind11::pybind11_fail("Could not allocate float object!");
    }
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator float() const { return (float)PyFloat_AsDouble(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator double() const { return (double)PyFloat_AsDouble(m_ptr); }

  static bool isFloat_(PyObject *o) {
    if (PyFloat_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.float32" || name == "numpy.float64") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isFloat_(o)) {
      ret = PyFloat_FromDouble(PyFloat_AsDouble(o));
    } else {
      pybind11::set_error(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

/// Extended python int object.
///
/// Includes `int`, `numpy.intXXX`.
class Int : public pybind11::object {
public:
  PYBIND11_OBJECT_CVT(Int, object, isInt_, convert_)

  // Allow implicit conversion from int:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Int(long value) : object(PyLong_FromLong((long)value), stolen_t{}) {
    if (!m_ptr) {
      pybind11::pybind11_fail("Could not allocate float object!");
    }
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int8_t() const { return (std::int8_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int16_t() const { return (std::int16_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int32_t() const { return (std::int32_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int64_t() const { return (std::int64_t)PyLong_AsLong(m_ptr); }

  static bool isInt_(PyObject *o) {
    if (PyLong_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.int8" || name == "numpy.int16" ||
        name == "numpy.int32" || name == "numpy.int64") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isInt_(o)) {
      ret = PyLong_FromLong(PyLong_AsLong(o));
    } else {
      pybind11::set_error(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

template <typename T>
inline char const *typeName() {
  return typeid(T).name();
}
template <>
inline char const *typeName<py_ext::Float>() {
  return "float";
}
template <>
inline char const *typeName<py_ext::Complex>() {
  return "complex";
}
template <>
inline char const *typeName<py_ext::Int>() {
  return "long";
}
template <>
inline char const *typeName<pybind11::int_>() {
  return "long";
}
template <>
inline char const *typeName<pybind11::bool_>() {
  return "bool";
}
template <>
inline char const *typeName<pybind11::list>() {
  return "list";
}

template <typename T, pybind11::detail::enable_if_t<
                          std::is_base_of<pybind11::object, T>::value, int> = 0>
inline bool isConvertible(pybind11::handle o) {
  return pybind11::isinstance<T>(o);
}
template <>
inline bool isConvertible<Complex>(pybind11::handle o) {
  return pybind11::isinstance<Complex>(o) || pybind11::isinstance<Float>(o) ||
         pybind11::isinstance<pybind11::int_>(o);
}
template <>
inline bool isConvertible<Float>(pybind11::handle o) {
  return pybind11::isinstance<Float>(o) ||
         pybind11::isinstance<pybind11::int_>(o);
}

template <typename T>
inline pybind11::object convert(T value) = delete;

template <>
inline pybind11::object convert(bool value) {
  return pybind11::bool_(value);
}

template <>
inline pybind11::object convert(std::int8_t value) {
  return pybind11::int_(value);
}

template <>
inline pybind11::object convert(std::int16_t value) {
  return pybind11::int_(value);
}

template <>
inline pybind11::object convert(std::int32_t value) {
  return pybind11::int_(value);
}

template <>
inline pybind11::object convert(std::int64_t value) {
  return pybind11::int_(value);
}

template <>
inline pybind11::object convert(float value) {
  return Float(value);
}

template <>
inline pybind11::object convert(double value) {
  return Float(value);
}

template <>
inline pybind11::object convert(std::complex<float> value) {
  return Complex(value);
}

template <>
inline pybind11::object convert(std::complex<double> value) {
  return Complex(value);
}

} // namespace py_ext
