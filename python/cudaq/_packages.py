# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from importlib.metadata import distribution
import os.path


def find_package_location(package_name):
    path = None
    try:
        path = _find_package_location_by_license(package_name)
    except:
        pass
    if path is None:
        path = _find_package_location_by_root(package_name)
    return path


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


def get_library_path(library, cuda_major=11):
    if library in ("cuda-runtime", "cublas", "cusolver", "cusparse"):
        package_name = f"nvidia-{library}-cu{cuda_major}"
        subdir = library.replace("-", "_")
    elif library in ("cutensor", "custatevec", "cutensornet"):
        package_name = f"{library}-cu{cuda_major}"
        subdir = ""
    else:
        raise NotImplementedError(f"library {library} is not recognized")

    dirname = os.path.join(find_package_location(package_name), subdir)
    assert os.path.isdir(dirname)
    return dirname


def get_include_path(library, cuda_major=11):
    dirname = os.path.join(get_library_path(library, cuda_major), "include")
    assert os.path.isdir(dirname)
    return dirname


def get_link_path(library, cuda_major=11):
    dirname = os.path.join(get_library_path(library, cuda_major), "lib")
    assert os.path.isdir(dirname)
    return dirname
