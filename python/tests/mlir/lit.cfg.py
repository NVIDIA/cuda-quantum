# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import subprocess
import sys
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

# CUDA-Q targets that this build actually produced. A target is only present
# when its backend was enabled at configure time (e.g. CUDAQ_ENABLE_OQC_BACKEND),
# so tests that pass `--target <name>` must gate on the corresponding feature
# with `// REQUIRES: cudaq-target-<name>` rather than assuming every target was
# built.
# This must be fatal rather than a warning: if enumeration fails no features
# are registered, and every test gated on one silently becomes "Unsupported"
# instead of running -- a green run that tested nothing.
_py_pkg_dir = os.path.join(os.path.dirname(config.cudaq_lib_dir), 'python')
# The configured interpreter may be gone by test time. scikit-build wheel
# builds configure in a temporary environment that is later deleted.
_python = config.python_executable
if not _python or not os.path.isfile(_python):
    _python = sys.executable
try:
    _targets = subprocess.check_output([
        _python, '-c',
        'import cudaq; print(" ".join(t.name for t in cudaq.get_targets()))'
    ],
                                       env=dict(os.environ,
                                                PYTHONPATH=_py_pkg_dir),
                                       stderr=subprocess.STDOUT,
                                       text=True).split()
except Exception as e:
    lit_config.fatal('Could not enumerate CUDA-Q targets, so cudaq-target-* '
                     'features cannot be set and gated tests would silently '
                     'be skipped: %s' % e)
if not _targets:
    lit_config.fatal('CUDA-Q reported no available targets; cudaq-target-* '
                     'gating would silently skip every gated test.')
for _t in _targets:
    config.available_features.add('cudaq-target-' + _t)

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
