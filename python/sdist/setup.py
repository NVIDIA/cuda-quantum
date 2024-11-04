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


release_version = os.getenv("CUDA_QUANTUM_VERSION", "0.0.0")
if release_version == '': release_version = "0.0.0"

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
    # Case 2: install sdist
    install_requires = [f"{_infer_best_package()}=={release_version}",]
    cmdclass = {}
    # FIXME: I don't think we need this? SEE https://discuss.python.org/t/wheel-caching-and-non-deterministic-builds/7687/6
    #cmdclass = {'bdist_wheel': bdist_wheel} if bdist_wheel is not None else {}


setup(
    name="cudaq",
    version=release_version,
    description="Python bindings for the CUDA-Q toolkit for heterogeneous quantum-classical workflows.",
    url="https://developer.nvidia.com/cuquantum-sdk",
    project_urls={
        "Homepage": "https://developer.nvidia.com/cuda-quantum",
        "Documentation": "https://nvidia.github.io/cuda-quantum",
        "Repository": "https://github.com/NVIDIA/cuda-quantum",
        "Releases": "https://nvidia.github.io/cuda-quantum/latest/releases.html",
    },
    author="NVIDIA Corporation & Affiliates",
    license_files=('LICENSE',),
    keywords=[ "cuda-quantum", "cuda", "quantum", "quantum computing", "nvidia", "high-performance computing" ],
    zip_safe=False,
    setup_requires=[
        "setuptools",
        "wheel",
    ],
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License 2.0 (Apache-2.0)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    cmdclass=cmdclass,
)
