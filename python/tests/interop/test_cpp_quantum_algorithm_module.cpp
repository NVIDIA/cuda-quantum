/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/qkernel.h"
#include "quantum_lib/quantum_lib.h"
#include "runtime/interop/PythonCppInterop.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

NB_MODULE(cudaq_test_cpp_algo, m) {
  // Example of how to expose C++ kernels.
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>>(
      m, "qstd", "qft", "(Fake) Quantum Fourier Transform.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "another", "Demonstrate we can have multiple ones.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, std::size_t>(
      m, "qstd", "uccsd", "");

  // Callback tests
  m.def(
      "run0",
      [](nanobind::object qern, std::size_t qnum) {
        cudaq::python::launch_specialized_py_decorator<cudaq::qkernel<void()>>(
            qern, cudaq::sit_and_spin_test, qnum);
      },
      "");
  m.def(
      "run0b",
      [](nanobind::object qern, std::size_t qnum) {
        // This idiom uses argument marshaling instead of specialization. This
        // allows `entryPoint` to be called with different arguments. Note that
        // the `decorator` must remain alive for `entryPoint` to be valid.
        cudaq::python::CppPyKernelDecorator decorator(qern);
        auto entryPoint =
            decorator.getDirectKernelCall<cudaq::qkernel<void(std::size_t)>>();
        marshal_test(std::move(entryPoint), qnum);
      },
      "");
  m.def(
      "run1",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<cudaq::qkernel<void()>>(
            qern, cudaq::plug_and_chug_test);
      },
      "");
  m.def(
      "run2",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<
            cudaq::qkernel<void(cudaq::qvector<> &)>>(qern,
                                                      cudaq::brain_bend_test);
      },
      "");
  m.def(
      "run3",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<
            cudaq::qkernel<void(cudaq::qvector<> &, std::size_t)>>(
            qern, cudaq::most_curious_test);
      },
      "");
  m.def(
      "run4",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<
            cudaq::qkernel<std::size_t(cudaq::qvector<> &, std::size_t)>>(
            qern, cudaq::callback_test);
      },
      "");

  m.def(
      "run5",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<
            cudaq::qkernel<std::vector<float>()>>(qern, cudaq::py_ret_test1);
      },
      "");

  m.def(
      "run6",
      [](nanobind::object qern) {
        cudaq::python::launch_specialized_py_decorator<
            cudaq::qkernel<std::vector<float>(std::size_t)>>(
            qern, cudaq::py_ret_test2);
      },
      "");
}
