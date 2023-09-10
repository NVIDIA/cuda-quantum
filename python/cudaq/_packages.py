# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from importlib.metadata import distribution
import os.path


def _find_package_location_by_root(package_name):
    dist = distribution(package_name)
    roots = set()
    for f in dist.files:
        dirname = os.path.dirname(str(f.locate()))
        if not dirname.endswith("dist-info") and not dirname.endswith(
                "__pycache__"):
            roots.add(dirname)
    path = os.path.commonprefix(tuple(roots))
    return path


def _find_package_location_by_license(package_name):
    dist = distribution(package_name)
    for f in dist.files:
        if str(f).endswith("LICENSE"):
            license = f
            break
    else:
        raise RuntimeError(f"cannot locate the directory for {package_name}")
    path = os.path.dirname(license.locate())
    return path


def get_library_path(package_name):
    subdir = ""
    if package_name.startswith("nvidia-"):
        subdir = "-".join(package_name.split("-")[1:-1])

    try:
        package_location = _find_package_location_by_license(package_name)
    except:
        package_location = _find_package_location_by_root(package_name)

    dirname = os.path.join(package_location, subdir, "lib")
    assert os.path.isdir(dirname)
    return dirname
