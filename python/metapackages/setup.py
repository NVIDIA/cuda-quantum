# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ctypes, os, sys
import importlib.util
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
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        retval = func(major, minor)
        version = major.value * 1000 + minor.value * 10
    else:
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
        ]
        version_ref = ctypes.c_int()
        retval = func(version_ref)
        version = version_ref.value

    if retval != 0:
        raise Exception(f'{libname}: {func} returned error: {retval}')
    _log(f'Detected version: {version}')
    return version


def _get_cuda_version() -> Optional[int]:
    """Returns the detected CUDA version or None."""

    version = None

    # First, try to get version from NVIDIA Management Library
    try:
        _log(f'Trying to detect CUDA version using NVIDIA Management Library')
        from pynvml import nvmlInit, nvmlSystemGetCudaDriverVersion
        nvmlInit()
        version = nvmlSystemGetCudaDriverVersion()
    except Exception as e:
        _log(f"Error: {e}")  # log and move on
    if version is not None:
        _log(f'Detected version: {version}')
        return version

    # Fall back to try and detect version from NVRTC
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

    # Last try to detect version from CUDART (a CUDA context will be initialized)
    libnames = [
        'libcudart.so.12',
        'libcudart.so.11.0',
    ]
    _log(f'Trying to detect CUDA version from libraries: {libnames}')
    try:
        version = _get_version_from_library(libnames, 'cudaRuntimeGetVersion',
                                            False)
    except Exception as e:
        _log(f"Error: {e}")  # log and move on
    if version is not None:
        return version

    _log("Autodetection failed")
    return None


def _infer_best_package() -> str:

    # Find the existing wheel installation
    installed = []
    for pkg_suffix in ['', '-cu11', '-cu12']:
        _log(f"Looking for existing installation of cuda-quantum{pkg_suffix}.")
        try:
            package_spec = importlib.util.find_spec(f"cuda-quantum{pkg_suffix}")
            if package_spec is None:
                _log("No installation found.")
            else:
                installed.append(f"cuda-quantum{pkg_suffix}")
                _log("Installation found.")
        except:
            _log("No installation found.")
            pass

    cuda_version = _get_cuda_version()
    if cuda_version is None:
        cudaq_bdist = 'cuda-quantum-cu12'
    elif cuda_version < 11000:
        raise Exception(f'Your CUDA version ({cuda_version}) is too old.')
    elif cuda_version < 12000:
        cudaq_bdist = 'cuda-quantum-cu11'
    elif cuda_version < 13000:
        cudaq_bdist = 'cuda-quantum-cu12'
    else:
        raise Exception(f'Your CUDA version ({cuda_version}) is too new.')
    _log(f"Identified {cudaq_bdist} as the best package.")

    # Disallow -cu11 & -cu12 wheels from coexisting
    conflicting = ", ".join((pkg for pkg in installed if pkg != cudaq_bdist))
    _log(f"Conflicting packages: {conflicting}")
    if conflicting != '':
        _log("Abort.")
        raise Exception(
            f'You have a conflicting CUDA-Q version installed.'
            f'Please remove the following package(s): {conflicting}')
    return cudaq_bdist


# This setup handles 3 cases:
#   1. At the release time, we use it to generate source distribution (which contains
#      this script).
#   2. If the source distribution is generated for the deprecated cuda-quantum package
#      name, this script raises an exception at install time asking to install cudaq.
#   3. If the source distribution is generated for a valid cudaq version,
#      this script identifies the installed CUDA version at cudaq install time
#      and downloads the corresponding CUDA-Q binary distribution.
# For case 1, `CUDAQ_META_WHEEL_BUILD` is set to 1.
setup_dir = os.path.dirname(os.path.abspath(__file__))
data_files = []
install_requires = []
if os.environ.get('CUDAQ_META_WHEEL_BUILD', '0') == '1':
    # Case 1: create source distribution
    if os.path.exists(os.path.join(setup_dir, "_deprecated.txt")):
        data_files = [('', [
            '_deprecated.txt',
        ])]  # extra files to be copied into the distribution
else:
    # Case 2: install cuda-quantum source distribution
    if os.path.exists(os.path.join(setup_dir, "_deprecated.txt")):
        with open(os.path.join(setup_dir, "_deprecated.txt"), "r") as f:
            deprecation = f.read()
        raise Exception(f'This package is deprecated. \n' + deprecation)
    # Case 3: install cudaq source distribution
    with open(os.path.join(setup_dir, "_version.txt"), "r") as f:
        __version__ = f.read()
    install_requires = [
        f"{_infer_best_package()}=={__version__}",
    ]

setup(
    zip_safe=False,
    data_files=data_files,
    install_requires=install_requires,
)
