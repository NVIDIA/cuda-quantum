# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import bisect

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# The name of this test suite.
config.name = 'CUDAQ'

# The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# A list of file extensions to treat as test files.
config.suffixes = ['.cpp', '.ll', '.mlir', '.qke']

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))
config.substitutions.append(('%llvmInclude', config.llvm_install + "/include"))
config.substitutions.append(('%cudaq_lib_dir', config.cudaq_lib_dir))
config.substitutions.append(('%cudaq_target_dir', config.cudaq_target_dir))
config.substitutions.append(('%cudaq_src_dir', config.cudaq_src_dir))
config.substitutions.append(('%cudaq_plugin_ext', config.cudaq_plugin_ext))

llvm_config.use_default_substitutions()

# Ask `llvm-config` about asserts
llvm_config.feature_config([('--assertion-mode', {'ON': 'asserts'})])

# Allow to require specific build targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

# Allow to filter tests based on environment variables
cpp_stds = ['c++17', 'c++20', 'c++23']
std_up_to = os.environ.get('CUDAQ_CPP_STD', 'c++23').lower()
for std in cpp_stds[:bisect.bisect(cpp_stds, std_up_to)]:
    config.available_features.add(std)
std_default = os.environ.get('CUDAQ_CPP_STD')
if std_default is None:
    config.substitutions.append(('%cpp_std', ''))
else:
    config.substitutions.append(('%cpp_std', '-std=' + std_default.lower()))

# Exclude a list of directories from the test suite:
#   - 'Inputs' contain auxiliary inputs for various tests.
local_excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']
config.excludes = [exclude for exclude in config.excludes] + local_excludes

# The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.cudaq_obj_root, 'test')

# Tweak the PATH to include the tools directory.
llvm_config.with_environment('PATH', config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
