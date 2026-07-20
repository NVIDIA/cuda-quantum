# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import subprocess
import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PYCUDAQMLIR'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))
config.substitutions.append(('%llvmInclude', config.llvm_install + "/include"))

llvm_config.use_default_substitutions()

# ask llvm-config about asserts
llvm_config.feature_config([('--assertion-mode', {'ON': 'asserts'})])

# Targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py',
    'random_gen.py'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.cudaq_obj_root, 'python/tests/mlir')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.cudaq_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Generate phase-folding tests
gen_tests_dir = os.path.join(config.cudaq_src_dir, 'python', 'tests', 'mlir',
                             'generated')
os.makedirs(gen_tests_dir, exist_ok=True)  # mode=0o777


def generate_phasefolding_test(filename, seed, min_block_length,
                               max_block_length, rz_weight):
    test_src_dir = os.path.join(config.cudaq_src_dir, 'python', 'tests', 'mlir',
                                'phase_folding')
    with open(os.path.join(gen_tests_dir, filename + str(seed) + '.py'),
              'w') as fout:
        subprocess.run([
            sys.executable, 'random_gen.py', filename + '.py.template',
            '--seed=' + str(seed), '--block-length=' + str(min_block_length) +
            '-' + str(max_block_length), '--rz-weight=' + str(rz_weight)
        ],
                       cwd=test_src_dir,
                       stdout=fout)


for seed in range(1, 11):
    generate_phasefolding_test('branch-in-loop', seed, 30, 45, 0.5)
for seed in range(1, 11):
    generate_phasefolding_test('loop-with-break', seed, 20, 30, 0.5)
generate_phasefolding_test('straight-line', 27, 100, 100, 0.5)
generate_phasefolding_test('subkernel', 1, 20, 30, 0.5)
