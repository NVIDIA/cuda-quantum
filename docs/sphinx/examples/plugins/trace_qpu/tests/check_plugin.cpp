/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: example-plugins
// RUN: rm -rf %t.xdg
// RUN: test -f %cudaq_example_plugins_dir/trace-qpu/targets/trace_qpu.yml
// RUN: test -f
// %cudaq_example_plugins_dir/trace-qpu/lib/libcudaq-qpu-trace_qpu%cudaq_plugin_ext
// RUN: FileCheck %s --check-prefix=TRACE-YAML
// --input-file=%cudaq_example_plugins_dir/trace-qpu/targets/trace_qpu.yml RUN:
// cudaq-quake %s | env LD_LIBRARY_PATH=%cudaq_lib_dir cudaq-opt
// --load-cudaq-plugin
// %cudaq_example_plugins_dir/trace-qpu/lib/libcudaq-qpu-trace_qpu%cudaq_plugin_ext
// --pass-pipeline="builtin.module(func.func(trace-qpu-summary))" -o
// %t.trace.mlir 2>&1 | FileCheck %s --check-prefix=SUMMARY RUN: env
// XDG_DATA_HOME=%t.xdg cudaq-install-plugin --copy
// %cudaq_example_plugins_dir/trace-qpu RUN: env XDG_DATA_HOME=%t.xdg
// cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST RUN: env
// XDG_DATA_HOME=%t.xdg nvq++ --list-targets | FileCheck %s
// --check-prefix=TARGETS RUN: PYTHONPATH=%cudaq_target_dir/../python python3 -c
// "import cudaq;
// cudaq.register_backend_path('%cudaq_example_plugins_dir/trace-qpu');
// cudaq.set_target('trace_qpu'); cudaq.reset_target()"

// TRACE-YAML: name: trace_qpu
// TRACE-YAML: platform-qpu: trace_qpu
// TRACE-YAML: jit-high-level-pipeline: "func.func(trace-qpu-summary)"
// TRACE-YAML: plugin-libraries:
// TRACE-YAML-NEXT: - libcudaq-qpu-trace_qpu

// SUMMARY: trace-qpu-summary: kernel={{.*}}trace_qpu_test_kernel{{.*}}
// quake_ops={{[0-9]+}} measurements=1

// LIST: user{{[[:space:]]+}}trace_qpu

// TARGETS: trace_qpu

#include <cudaq.h>

__qpu__ void trace_qpu_test_kernel() {
  cudaq::qubit q;
  h(q);
  mz(q);
}

int main() { return 0; }
