/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: rm -rf %t.user %cudaq_target_dir/../plugins/nvqpp-b5-system-only %cudaq_target_dir/../plugins/nvqpp-b5-precedence
// RUN: mkdir -p %t.user/cudaq/plugins/nvqpp-b5-user-only/targets %t.user/cudaq/plugins/nvqpp-b5-user-only/lib %t.user/cudaq/plugins/nvqpp-b5-user-only/data
// RUN: mkdir -p %t.user/cudaq/plugins/nvqpp-b5-precedence/targets %t.user/cudaq/plugins/nvqpp-b5-precedence/lib
// RUN: mkdir -p %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets %cudaq_target_dir/../plugins/nvqpp-b5-system-only/lib
// RUN: mkdir -p %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets %cudaq_target_dir/../plugins/nvqpp-b5-precedence/lib
// RUN: echo "topology" > %t.user/cudaq/plugins/nvqpp-b5-user-only/data/topology.txt
// RUN: echo 'name: nvqpp-b5-user-only' > %t.user/cudaq/plugins/nvqpp-b5-user-only/targets/nvqpp-b5-user-only.yml
// RUN: echo 'description: "B5 user plugin target."' >> %t.user/cudaq/plugins/nvqpp-b5-user-only/targets/nvqpp-b5-user-only.yml
// RUN: echo 'config:' >> %t.user/cudaq/plugins/nvqpp-b5-user-only/targets/nvqpp-b5-user-only.yml
// RUN: echo '  library-mode: true' >> %t.user/cudaq/plugins/nvqpp-b5-user-only/targets/nvqpp-b5-user-only.yml
// RUN: echo '  preprocessor-defines: ["-DB5_USER_ONLY=%PLUGIN_ROOT%/data/topology.txt"]' >> %t.user/cudaq/plugins/nvqpp-b5-user-only/targets/nvqpp-b5-user-only.yml
// RUN: echo 'name: nvqpp-b5-system-only' > %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets/nvqpp-b5-system-only.yml
// RUN: echo 'description: "B5 system plugin target."' >> %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets/nvqpp-b5-system-only.yml
// RUN: echo 'config:' >> %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets/nvqpp-b5-system-only.yml
// RUN: echo '  library-mode: true' >> %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets/nvqpp-b5-system-only.yml
// RUN: echo '  preprocessor-defines: ["-DB5_SYSTEM_ONLY"]' >> %cudaq_target_dir/../plugins/nvqpp-b5-system-only/targets/nvqpp-b5-system-only.yml
// RUN: echo 'name: nvqpp-b5-precedence' > %t.user/cudaq/plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo 'description: "B5 user precedence target."' >> %t.user/cudaq/plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo 'config:' >> %t.user/cudaq/plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo '  library-mode: true' >> %t.user/cudaq/plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo '  preprocessor-defines: ["-DB5_FROM_USER"]' >> %t.user/cudaq/plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo 'name: nvqpp-b5-precedence' > %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo 'description: "B5 system fallback target."' >> %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo 'config:' >> %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo '  library-mode: true' >> %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: echo '  preprocessor-defines: ["-DB5_FROM_SYSTEM"]' >> %cudaq_target_dir/../plugins/nvqpp-b5-precedence/targets/nvqpp-b5-precedence.yml
// RUN: env XDG_DATA_HOME=%t.user nvq++ --list-targets | FileCheck %s --check-prefix=LIST
// RUN: env XDG_DATA_HOME=%t.user nvq++ -v --target nvqpp-b5-user-only %s -o %t.user.exe 2>&1 | FileCheck %s --check-prefix=USER
// RUN: env XDG_DATA_HOME=%t.user nvq++ -v --target nvqpp-b5-system-only %s -o %t.system.exe 2>&1 | FileCheck %s --check-prefix=SYSTEM
// RUN: env XDG_DATA_HOME=%t.user nvq++ -v --target nvqpp-b5-precedence %s -o %t.precedence.exe 2>&1 | FileCheck %s --check-prefix=PRECEDENCE

// LIST-DAG: nvqpp-b5-user-only
// LIST-DAG: nvqpp-b5-system-only

// USER: -DB5_USER_ONLY=
// USER-SAME: /cudaq/plugins/nvqpp-b5-user-only/data/topology.txt
// USER: -Wl,-rpath,
// USER-SAME: /cudaq/plugins/nvqpp-b5-user-only/lib
// USER: -L
// USER-SAME: /cudaq/plugins/nvqpp-b5-user-only/lib

// SYSTEM: -DB5_SYSTEM_ONLY
// SYSTEM: -Wl,-rpath,
// SYSTEM-SAME: /plugins/nvqpp-b5-system-only/lib
// SYSTEM: -L
// SYSTEM-SAME: /plugins/nvqpp-b5-system-only/lib

// PRECEDENCE: -DB5_FROM_USER
// PRECEDENCE-NOT: -DB5_FROM_SYSTEM

int main() { return 0; }
