/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

/// Additional type checking helpers for python.
class conversion {
  public:
    static bool isComplex(py::handle h) {
      return isComplex_(h.ptr());
    }

    static bool isFloat(py::handle h) {
      return isFloat_(h.ptr());
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
};

/// Complex object.
class complex_ : public py::object {
  public:
    // Python does not provide a `PyNumber_Complex` function, so we provide our own `convert`.
    PYBIND11_OBJECT_CVT(complex_, object, PyComplex_Check, convert)

    complex_(double real, double imag): object(PyComplex_FromDoubles(real, imag), stolen_t{}) {}

    complex_(std::complex<double> value) : complex_((double) value.real(), (double) value.imag()) {}

    complex_(std::complex<float> value) : complex_((double) value.real(), (double) value.imag()) {}

    operator std::complex<double>() {
      auto value = PyComplex_AsCComplex(m_ptr);
      return std::complex<double>(value.real, value.imag);
    }
    operator std::complex<float>() {
      auto value = PyComplex_AsCComplex(m_ptr);
      return std::complex<float>(value.real, value.imag);
    }

  private:
    static PyObject *convert(PyObject *o) {
      PyObject *ret = nullptr;
      if(conversion::isComplex_(o)) {
        double real = PyComplex_RealAsDouble(o);
        double imag = PyComplex_ImagAsDouble(o);
        ret = PyComplex_FromDoubles(real, imag);
      } else {
        py::set_error(PyExc_TypeError, "Unexpected type");
      }
      return ret;
    }
};
