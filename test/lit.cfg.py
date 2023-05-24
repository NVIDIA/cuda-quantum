# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

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
config.substitutions.append(('%cudaq_plugin_ext', config.cudaq_plugin_ext))

llvm_config.use_default_substitutions()

# Ask `llvm-config` about asserts
llvm_config.feature_config([('--assertion-mode', {'ON': 'asserts'})])

config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

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
