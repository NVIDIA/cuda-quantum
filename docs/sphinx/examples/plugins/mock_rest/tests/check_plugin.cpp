/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: rm -rf /tmp/cudaq-lit-mock-rest-xdg
// RUN: test -f %cudaq_example_plugins_dir/mock-rest/targets/mock_rest.yml
// RUN: test -f %cudaq_example_plugins_dir/mock-rest/lib/libcudaq-serverhelper-mock_rest%cudaq_plugin_ext
// RUN: FileCheck %s --check-prefix=MOCK-YAML --input-file=%cudaq_example_plugins_dir/mock-rest/targets/mock_rest.yml
// RUN: env XDG_DATA_HOME=/tmp/cudaq-lit-mock-rest-xdg cudaq-install-plugin --copy %cudaq_example_plugins_dir/mock-rest
// RUN: env XDG_DATA_HOME=/tmp/cudaq-lit-mock-rest-xdg cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST
// RUN: env XDG_DATA_HOME=/tmp/cudaq-lit-mock-rest-xdg nvq++ --list-targets | FileCheck %s --check-prefix=TARGETS
// RUN: PYTHONPATH=%cudaq_target_dir/../python python3 -c "import cudaq; cudaq.register_backend_path('%cudaq_example_plugins_dir/mock-rest'); cudaq.set_target('mock_rest'); cudaq.reset_target()"
// RUN: rm -rf %t.python %t.pip-cache %t.python-xdg
// RUN: env PIP_CACHE_DIR=%t.pip-cache python3 -m pip install --no-deps --no-build-isolation --target %t.python %cudaq_example_plugins_dir/mock-rest
// RUN: env PYTHONPATH=%t.python:%cudaq_target_dir/../python python3 -c "import cudaq; assert cudaq.has_target('mock_rest'); cudaq.set_target('mock_rest'); cudaq.reset_target()"
// RUN: env PYTHONPATH=%t.python XDG_DATA_HOME=%t.python-xdg python3 -m cudaq_example_mock_rest --install-nvqpp
// RUN: test -L %t.python-xdg/cudaq/plugins/mock-rest
// RUN: env XDG_DATA_HOME=%t.python-xdg nvq++ --list-targets | FileCheck %s --check-prefix=TARGETS
// clang-format on

// MOCK-YAML: name: mock_rest
// MOCK-YAML: platform-qpu: remote_rest
// MOCK-YAML-NOT: plugin-libraries

// LIST: user{{[[:space:]]+}}mock-rest

// TARGETS: mock_rest

#include <cudaq.h>

__qpu__ void mock_rest_test_kernel() {
  cudaq::qubit q;
  h(q);
  mz(q);
}

int main() { return 0; }
