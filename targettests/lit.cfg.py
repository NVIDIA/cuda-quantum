# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import shutil
import bisect

import lit.util
from lit.llvm import llvm_config
import lit.formats

# The name of this test suite.
config.name = 'CUDAQ-Target'

# `testFormat`: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = ['.cpp', '.config']

# Exclude a list of directories from the test suite:
#   - 'Inputs' contain auxiliary inputs for various tests.
local_excludes = ['ionq', 'iqm', 'oqc', 'quantinuum',
                  'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']
config.excludes = [exclude for exclude in config.excludes] + local_excludes

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))
config.substitutions.append(('%llvmInclude', config.llvm_install + "/include"))
config.substitutions.append(('%cudaq_lib_dir', config.cudaq_lib_dir))
config.substitutions.append(('%cudaq_plugin_ext', config.cudaq_plugin_ext))
config.substitutions.append(('%cudaq_target_dir', config.cudaq_target_dir))
config.substitutions.append(('%cudaq_src_dir', config.cudaq_src_dir))
config.substitutions.append(('%iqm_test_src_dir', config.cudaq_src_dir + "/runtime/cudaq/platform/default/rest/helpers/iqm"))

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

# The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.cudaq_obj_root, 'targettests')

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Tweak the PATH to include the tools directory.
llvm_config.with_environment('PATH', config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
