/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// `dem_from_kernel` runs on the default execution manager, so this module links
// cudaq-em-default. Keep it separate from cudaq_test_cpp_algo, which checks
// that kernels compiled with `nvq++ -c` can be called without it.

#include "quantum_lib/quantum_lib.h"
#include "runtime/interop/PythonCppInterop.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/qis/qkernel.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

NB_MODULE(cudaq_test_cpp_dem, m) {
  m.def(
      "run_measure_handle_callback",
      [](nanobind::object qern) {
        cudaq::python::CppPyKernelDecorator decorator(qern);
        auto entryPoint = decorator.getDirectKernelCall<cudaq::qkernel<
            std::vector<cudaq::measure_handle>(cudaq::qvector<> &)>>();
        cudaq::M2DSparseMatrix m2d;
        cudaq::M2OSparseMatrix m2o;
        auto dem = cudaq::dem_from_kernel(cudaq::measure_handle_callback_test,
                                          m2d, m2o, entryPoint);
        return nanobind::make_tuple(dem, m2d.rows, m2d.num_measurements);
      },
      "");
}
