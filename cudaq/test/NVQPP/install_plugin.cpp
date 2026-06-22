/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: rm -rf %t.install %t.xdg %cudaq_target_dir/../plugins/b6-system-plugin
// RUN: mkdir -p %t.install/b6-user-plugin/targets %t.install/b6-user-plugin/lib
// RUN: mkdir -p %t.install/b6-copy-plugin/targets %t.install/b6-copy-plugin/lib
// RUN: mkdir -p %t.install/b6-system-plugin/targets %t.install/b6-system-plugin/lib
// RUN: echo 'name: b6-user-target' > %t.install/b6-user-plugin/targets/b6-user-target.yml
// RUN: echo 'config:' >> %t.install/b6-user-plugin/targets/b6-user-target.yml
// RUN: echo '  library-mode: true' >> %t.install/b6-user-plugin/targets/b6-user-target.yml
// RUN: echo 'name: b6-copy-target' > %t.install/b6-copy-plugin/targets/b6-copy-target.yml
// RUN: echo 'config:' >> %t.install/b6-copy-plugin/targets/b6-copy-target.yml
// RUN: echo '  library-mode: true' >> %t.install/b6-copy-plugin/targets/b6-copy-target.yml
// RUN: echo 'name: b6-system-target' > %t.install/b6-system-plugin/targets/b6-system-target.yml
// RUN: echo 'config:' >> %t.install/b6-system-plugin/targets/b6-system-target.yml
// RUN: echo '  library-mode: true' >> %t.install/b6-system-plugin/targets/b6-system-target.yml
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin %t.install/b6-user-plugin
// RUN: test -L %t.xdg/cudaq/plugins/b6-user-plugin
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST-USER
// RUN: env XDG_DATA_HOME=%t.xdg nvq++ --list-targets | FileCheck %s --check-prefix=NVQPP-USER
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --copy %t.install/b6-copy-plugin
// RUN: test -d %t.xdg/cudaq/plugins/b6-copy-plugin
// RUN: not test -L %t.xdg/cudaq/plugins/b6-copy-plugin
// RUN: test -f %t.xdg/cudaq/plugins/b6-copy-plugin/targets/b6-copy-target.yml
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --system %t.install/b6-system-plugin
// RUN: test -L %cudaq_target_dir/../plugins/b6-system-plugin
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --list | FileCheck %s --check-prefix=LIST-BOTH
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --uninstall b6-user-plugin
// RUN: not test -L %t.xdg/cudaq/plugins/b6-user-plugin
// RUN: env XDG_DATA_HOME=%t.xdg cudaq-install-plugin --uninstall b6-system-plugin
// RUN: not test -L %cudaq_target_dir/../plugins/b6-system-plugin
// RUN: mkdir -p %t.install/not-a-plugin
// RUN: not env XDG_DATA_HOME=%t.xdg cudaq-install-plugin %t.install/not-a-plugin 2>&1 | FileCheck %s --check-prefix=INVALID

// LIST-USER: user{{[[:space:]]+}}b6-user-plugin
// NVQPP-USER: b6-user-target
// LIST-BOTH-DAG: user{{[[:space:]]+}}b6-user-plugin
// LIST-BOTH-DAG: user{{[[:space:]]+}}b6-copy-plugin
// LIST-BOTH-DAG: system{{[[:space:]]+}}b6-system-plugin
// INVALID: plugin root must contain a targets directory

int main() { return 0; }
