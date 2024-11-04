# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ctypes, pkg_resources, os, sys
from setuptools import setup
from typing import Optional


def _log(msg: str) -> None:
    sys.stdout.write(f'[cudaq] {msg}\n')
    sys.stdout.flush()

def _get_version_from_library(
        libnames: list[str],
        funcname: str,
        nvrtc: bool = False,
) -> Optional[int]:
    """Returns the library version from list of candidate libraries."""

    for libname in libnames:
        try:
            _log(f'Looking for library: {libname}')
            runtime_so = ctypes.CDLL(libname)
            break
        except Exception as e:
            _log(f'Failed to open {libname}: {e}')
    else:
        _log('No more candidate library to find')
        return None

    func = getattr(runtime_so, funcname, None)
    if func is None:
        raise Exception(f'{libname}: {func} could not be found')
    func.restype = ctypes.c_int

    if nvrtc:
        # nvrtcVersion
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        retval = func(major, minor)
        version = major.value * 1000 + minor.value * 10
    else:
        # cudaRuntimeGetVersion
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
        ]
        version_ref = ctypes.c_int()
        retval = func(version_ref)
        version = version_ref.value

    if retval != 0:  # NVRTC_SUCCESS or cudaSuccess
        raise Exception(f'{libname}: {func} returned error: {retval}')
    _log(f'Detected version: {version}')
    return version

def _get_cuda_version() -> Optional[int]:
    """Returns the detected CUDA version or None."""

    version = None

    # First try NVRTC
    libnames = [
        'libnvrtc.so.12',
        'libnvrtc.so.11.2',
        'libnvrtc.so.11.1',
        'libnvrtc.so.11.0',
    ]
    _log(f'Trying to detect CUDA version from libraries: {libnames}')
    try:
        version = _get_version_from_library(libnames, 'nvrtcVersion', True)
    except Exception as e:
        _log(f"Error: {e}")  # log and move on
    if version is not None:
        return version

    # Next try CUDART (side-effect: a CUDA context will be initialized)
    libnames = [
        'libcudart.so.12',
        'libcudart.so.11.0',
    ]
    _log(f'Trying to detect CUDA version from libraries: {libnames}')
    try:
        version = _get_version_from_library(libnames, 'cudaRuntimeGetVersion', False)
    except Exception as e:
        _log(f"Error: {e}")  # log and move on
    if version is not None:
        return version

    _log("Autodetection failed")
    return None

def _infer_best_package() -> str:

    # Find the existing wheel installation
    installed = []
    for cuda_major in ['11', '12']:
        try:
            pkg_resources.get_distribution(f"cuda-quantum-cu{cuda_major}")
            installed.append(f"cuda-quantum-cu{cuda_major}")
        except pkg_resources.DistributionNotFound:
            pass
    
    cuda_version = _get_cuda_version()
    if cuda_version is None:
        cudaq_bdist='cuda-quantum-cu12'
    elif cuda_version < 11000:
        raise Exception(f'Your CUDA version ({cuda_version}) is too old.')
    elif cuda_version < 12000:
        cudaq_bdist='cuda-quantum-cu11'
    elif cuda_version < 13000:
        cudaq_bdist='cuda-quantum-cu12'
    else:
        raise Exception(f'Your CUDA version ({cuda_version}) is too new.')

    # Disallow -cu11 & -cu12 wheels from coexisting
    conflicting=", ".join((pkg for pkg in installed if pkg != cudaq_bdist))
    if conflicting != '':
        raise Exception(
            f'You have a conflicting CUDA-Q version installed.'
            f'Please remove the following package(s): {conflicting}')
    return cudaq_bdist


# This setup.py handles 2 cases:
#   1. At the release time, we use it to generate sdist (which contains this script)
#   2. At the install time, this script identifies the installed CUDA version 
#      and downloads the corresponding CUDA-Q binary distribution
# For case 1, CUDAQ_META_WHEEL_BUILD is set to 1.
if os.environ.get('CUDAQ_META_WHEEL_BUILD', '0') == '1':
    # Case 1: generate sdist
    install_requires = []
    cmdclass = {}
else:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"_version.txt")) as f:
        __version__ = f.read()
    install_requires = [f"{_infer_best_package()}=={__version__}",]
    cmdclass = {}
    # FIXME: I don't think we need this? SEE https://discuss.python.org/t/wheel-caching-and-non-deterministic-builds/7687/6
    #cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else {}


setup(
    zip_safe=False,
    install_requires=install_requires,
)
