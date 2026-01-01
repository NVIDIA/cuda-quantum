# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, sys
import subprocess
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
local_excludes = ['Inputs']
config.excludes = [exclude for exclude in config.excludes] + local_excludes

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))
config.substitutions.append(('%llvmInclude', config.llvm_install + "/include"))
config.substitutions.append(('%cudaq_lib_dir', config.cudaq_lib_dir))
config.substitutions.append(('%cudaq_plugin_ext', config.cudaq_plugin_ext))
config.substitutions.append(('%cudaq_target_dir', config.cudaq_target_dir))
config.substitutions.append(('%cudaq_src_dir', config.cudaq_src_dir))
config.substitutions.append(('%iqm_tests_dir', config.cudaq_src_dir + "/targettests/Target/IQM"))

llvm_config.use_default_substitutions()

# Ask `llvm-config` about asserts
llvm_config.feature_config([('--assertion-mode', {'ON': 'asserts'})])

# Allow to require specific build targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

# The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# The root path where tests should be run.
config.test_exec_root = os.path.join(config.cudaq_obj_root, 'targettests')

# Propagate some variables from the host environment.
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Tweak the PATH to include the tools directory.
llvm_config.with_environment('PATH', config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Generate test cases

gen_tests_dir = os.path.join(config.cudaq_src_dir, 'targettests', 'generated', 'phase-folding')
os.makedirs(gen_tests_dir, exist_ok=True) # mode=0o777
def generate_phasefolding_test(filename, seed, min_block_length, max_block_length, rz_weight):
    test_src_dir = os.path.join(config.cudaq_src_dir, 'targettests', 'Remote-Sim', 'phase-folding')
    with open(os.path.join(gen_tests_dir, filename + str(seed) + '.cpp'), 'w') as fout:
        subprocess.run([sys.executable, 'random_gen.py', filename + '.template', '--seed=' + str(seed), '--block-length=' + str(min_block_length) + '-' + str(max_block_length), '--rz-weight=' + str(rz_weight)], cwd=test_src_dir, stdout=fout)
for seed in range(1, 11):
    generate_phasefolding_test('branch-in-loop', seed, 30, 45, 0.5)
for seed in range(1, 11):
    generate_phasefolding_test('loop-with-break', seed, 20, 30, 0.5)
generate_phasefolding_test('straight-line', 27, 100, 100, 0.5)
generate_phasefolding_test('subkernel', 1, 20, 30, 0.5)
