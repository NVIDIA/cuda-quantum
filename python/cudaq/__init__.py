# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import os

from importlib.metadata import distribution
import os.path


def _find_package_location_by_root(package_name):
    """This should not fail, unless the package is not installed."""
    dist = distribution(package_name)
    roots = set()
    for f in dist.files:
        dirname = os.path.dirname(str(f.locate()))
        if not dirname.endswith("dist-info") and not dirname.endswith("__pycache__"):
            roots.add(dirname)
    path = os.path.commonprefix(tuple(roots))
    return path


def _find_package_location_by_license(package_name):
    """This function assumes a file named LICENSE is placed at the package root."""
    dist = distribution(package_name)
    for f in dist.files:
        if str(f).endswith("LICENSE"):
            license = f
            break
    else:
        raise RuntimeError(f"cannot locate the directory for {package_name}")
    path = os.path.dirname(license.locate())
    return path


def _get_custatevec_libpath():
    package_name = "custatevec-cu11"
    try: package_loc = _find_package_location_by_license(package_name)
    except: package_loc = _find_package_location_by_root(package_name)
    dirname = os.path.join(package_loc, "lib")
    assert os.path.isdir(dirname)
    return dirname


# Find the custatevec library included in the cuquantum dependency.
# This isn't going to be visible in the user environment, 
# but it is visible to the application code that imports cudaq.
# Alternatively, we could pass it to initialize_cudaq.
os.environ["CUSTATEVEC_DYNLIBS"] = os.path.join(_get_custatevec_libpath(), "libcustatevec.so.1")


from ._pycudaq import *
from .domains import chemistry

initKwargs = {'target': 'default'}

if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]

if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]

initialize_cudaq(**initKwargs)