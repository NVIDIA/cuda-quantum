# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os
import bisect

import lit.formats

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "NVQPP-Unit"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(".", "")

config.test_format.test_suffixes = [
    "test_mqpu",
    "test_mpi",
    "test_gpu_get_state",
    "test_spin",
    "test_operators",
    "test_dynamics",
    "test_qudit",
    "test_photonics",
    "test_utils",
    "test_domains",
    "test_runtime_qpp",
    "test_runtime_dm",
    "test_runtime_sm",
]

# A list of file extensions to treat as test files.

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%llvmshlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%pluginext", config.llvm_plugin_ext))
config.substitutions.append(("%llvmInclude", config.llvm_install + "/include"))
config.substitutions.append(("%cudaq_lib_dir", config.cudaq_lib_dir))
config.substitutions.append(("%cudaq_target_dir", config.cudaq_target_dir))
config.substitutions.append(("%cudaq_src_dir", config.cudaq_src_dir))
config.substitutions.append(("%cudaq_plugin_ext", config.cudaq_plugin_ext))

llvm_config.use_default_substitutions()

# Ask `llvm-config` about asserts
llvm_config.feature_config([("--assertion-mode", {"ON": "asserts"})])

# Allow to require specific build targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

# The root path where tests are located.
config.test_source_root = os.path.join(config.cudaq_obj_root, "unittests")

# The root path where tests should be run.
config.test_exec_root = config.test_source_root

# Tweak the PATH to include the tools directory.
llvm_config.with_environment("PATH", config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
