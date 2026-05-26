# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import subprocess
import sys

import pytest


def _run_in_subprocess(code):
    """Run Python code in a fresh process so lazy import state is clean."""
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True)
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\n{result.stderr}")
    return result.stdout


def test_lazy_modules_not_eagerly_imported():
    """Verify that dynamics, kernels, and domains are not imported at
    `import cudaq` time."""
    _run_in_subprocess("""
import cudaq
import sys

for mod in ['cudaq.dynamics', 'cudaq.dynamics.schedule',
            'cudaq.dynamics.integrators', 'cudaq.kernels.uccsd',
            'cudaq.domains.chemistry', 'cudaq.dbg.ast']:
    assert mod not in sys.modules, f'{mod} was eagerly imported'
""")


def test_lazy_attrs_resolve():
    """Verify that lazy-loaded public API names resolve on first access."""
    _run_in_subprocess("""
import cudaq

# _LAZY_ATTRS
assert callable(cudaq.evolve)
assert callable(cudaq.evolve_async)
assert cudaq.Schedule is not None
assert cudaq.IntermediateResultSave is not None

# _LAZY_SUBMODULES
assert hasattr(cudaq.chemistry, '__name__')
assert hasattr(cudaq.uccsd, '__name__')
assert hasattr(cudaq.ast, '__name__')

# _DEFERRED_STAR_MODULES (integrator classes)
assert cudaq.RungeKuttaIntegrator is not None
assert cudaq.ScipyZvodeIntegrator is not None
""")


def test_lazy_attrs_cached_after_access():
    """Verify that lazy-loaded names are cached in globals after first access."""
    _run_in_subprocess("""
import cudaq

assert 'Schedule' not in vars(cudaq)
_ = cudaq.Schedule
assert 'Schedule' in vars(cudaq)
""")


def test_dir_includes_lazy_names():
    """Verify that `dir(cudaq)` includes lazy-loaded names for
    tab-completion."""
    _run_in_subprocess("""
import cudaq

d = dir(cudaq)
for name in ['evolve', 'evolve_async', 'Schedule',
             'IntermediateResultSave', 'chemistry', 'uccsd', 'ast',
             'RungeKuttaIntegrator', 'ScipyZvodeIntegrator']:
    assert name in d, f'{name} missing from dir(cudaq)'
""")


def test_unknown_attr_raises():
    """Verify that accessing a nonexistent attribute raises AttributeError."""
    _run_in_subprocess("""
import cudaq

try:
    cudaq.nonexistent_attribute_xyz
    assert False, 'should have raised'
except AttributeError:
    pass
""")
