/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: rm -rf %t.xdg
// RUN: test -f %cudaq_example_plugins_dir/mock-observe-qpu/targets/mock_observe_qpu.yml
// RUN: test -f %cudaq_example_plugins_dir/mock-observe-qpu/lib/libcudaq-mock_observe_qpu-qpu%cudaq_plugin_ext
// RUN: FileCheck %s --check-prefix=MOCK-YAML --input-file=%cudaq_example_plugins_dir/mock-observe-qpu/targets/mock_observe_qpu.yml
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --copy %cudaq_example_plugins_dir/mock-observe-qpu
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST
// RUN: env XDG_DATA_HOME=%t.xdg nvq++ --list-targets | FileCheck %s --check-prefix=TARGETS
// RUN: env XDG_DATA_HOME=%t.xdg nvq++ --target mock_observe_qpu %s -o %t.native
// RUN: env XDG_DATA_HOME=%t.xdg %t.native
// RUN: PYTHONPATH=%cudaq_target_dir/../python python3 -c "import cudaq; cudaq.register_backend_path('%cudaq_example_plugins_dir/mock-observe-qpu'); cudaq.set_target('mock_observe_qpu'); cudaq.reset_target()"
// RUN: rm -rf %t.python %t.pip-cache %t.python-xdg
// RUN: env PIP_CACHE_DIR=%t.pip-cache python3 -m pip install --no-deps --no-build-isolation --target %t.python %cudaq_example_plugins_dir/mock-observe-qpu
// RUN: env PYTHONPATH=%t.python:%cudaq_target_dir/../python python3 -c "import cudaq; assert cudaq.has_target('mock_observe_qpu'); cudaq.set_target('mock_observe_qpu'); cudaq.reset_target()"
// RUN: env PYTHONPATH=%t.python:%cudaq_target_dir/../python XDG_DATA_HOME=%t.python-xdg python3 -m cudaq_example_mock_observe_qpu --install-nvqpp
// RUN: test -L %t.python-xdg/cudaq/plugins/mock-observe-qpu
// RUN: env XDG_DATA_HOME=%t.python-xdg nvq++ --list-targets | FileCheck %s --check-prefix=TARGETS
// clang-format on

// MOCK-YAML: name: mock_observe_qpu
// MOCK-YAML: cudaq-version:
// MOCK-YAML: platform-qpu: mock_observe_qpu
// MOCK-YAML-NOT: plugin-libraries
// MOCK-YAML-NOT: observe-mode
// MOCK-YAML-NOT: remote_rest

// LIST: user{{[[:space:]]+}}mock-observe-qpu

// TARGETS: mock_observe_qpu

#include <cudaq.h>

__qpu__ void mock_observe_qpu_test_kernel() {
  cudaq::qubit q;
  h(q);
  mz(q);
}

int main() { return cudaq::get_platform().is_remote() ? 0 : 1; }
