# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# The name of this test suite.
config.name = 'RUNTIME'

# The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Each test under Regress/ is a "hybrid" test: a compiled executable (see
# add_cudaq_hybrid_test in CMakeLists.txt) whose own RUN line, embedded in a
# leading comment, invokes itself by name and pipes the output through
# FileCheck. Only the .cpp source carries that RUN line, so it is the only
# suffix lit needs to look for.
config.suffixes = ['.cpp']

llvm_config.use_default_substitutions()

# Exclude non-test files from the test suite.
local_excludes = ['CMakeLists.txt']
config.excludes = [exclude for exclude in config.excludes] + local_excludes

# The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.runtime_obj_root, 'runtime', 'test')

# Tweak the PATH to include the tools directory so the hybrid test
# executables built by CMake (and FileCheck/not) can be found.
llvm_config.with_environment('PATH', config.runtime_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
