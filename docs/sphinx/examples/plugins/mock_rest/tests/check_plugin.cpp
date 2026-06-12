/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: example-plugins
// RUN: rm -rf %t.xdg
// RUN: test -f %cudaq_example_plugins_dir/mock-rest/targets/mock_rest.yml
// RUN: test -f
// %cudaq_example_plugins_dir/mock-rest/lib/libcudaq-serverhelper-mock_rest%cudaq_plugin_ext
// RUN: FileCheck %s --check-prefix=MOCK-YAML
// --input-file=%cudaq_example_plugins_dir/mock-rest/targets/mock_rest.yml RUN:
// env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --copy
// %cudaq_example_plugins_dir/mock-rest RUN: env XDG_DATA_HOME=%t.xdg
// cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST RUN: env
// XDG_DATA_HOME=%t.xdg nvq++ --list-targets | FileCheck %s
// --check-prefix=TARGETS RUN: PYTHONPATH=%cudaq_target_dir/../python python3 -c
// "import cudaq;
// cudaq.register_backend_path('%cudaq_example_plugins_dir/mock-rest');
// cudaq.set_target('mock_rest'); cudaq.reset_target()"

// MOCK-YAML: name: mock_rest
// MOCK-YAML: platform-qpu: remote_rest
// MOCK-YAML-NOT: plugin-libraries

// LIST: user{{[[:space:]]+}}mock_rest

// TARGETS: mock_rest

#include <cudaq.h>

__qpu__ void mock_rest_test_kernel() {
  cudaq::qubit q;
  h(q);
  mz(q);
}

int main() { return 0; }
