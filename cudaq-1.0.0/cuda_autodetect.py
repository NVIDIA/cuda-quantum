import os
import sys
import ctypes
from typing import List
import pkg_resources

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

PACKAGE_NAME = 'cudaq'
PACKAGE_SUPPORTED_CUDA_VER = ['11', '12']
PACKAGE_RESOLUTION = None
CUDA_RESOLUTION = None

class AutoDetectionFailed(Exception):
    def __str__(self) -> str:
        return f'''
\n\n============================================================
{super().__str__()}
============================================================\n
'''
    
def _log(msg):
    sys.stdout.write(f'[{PACKAGE_NAME}] {msg}\n')
    sys.stdout.flush()

def _get_version_from_cuda_header():
    """
    Tries to extract the CUDA version from the /usr/local/cuda/include/cuda.h
    file. The format of CUDA_VERSION is a number like 11080, where 11080 means
    CUDA 11.8. We will divide by 1000 to get the major version.
    """
    cuda_header = '/usr/local/cuda/include/cuda.h'
    if os.path.exists(cuda_header):
        try:
            with open(cuda_header, 'r') as f:
                for line in f:
                    if '#define CUDA_VERSION' in line:
                        version_str = line.split()[-1].strip()
                        version_int = int(version_str)
                        major_version = version_int / 1000
                        _log(f'Detected CUDA version {major_version} from cuda.h (raw: {version_str})')
                        return version_int
        except Exception as e:
            _log(f"Error reading {cuda_header}: {e}")
    _log(f"No CUDA version found in cuda.h")
    return None

def _get_version_from_library(libnames, funcname, has_major_minor=False):
    for libname in libnames:
        try:
            _log(f'Looking for library: {libname}')
            runtime_so = ctypes.CDLL(libname)
            break
        except Exception as e:
            _log(f'Failed to open {libname}: {e}')
    else:
        _log('No more candidate libraries to find')
        return None

    func = getattr(runtime_so, funcname, None)
    if func is None:
        raise AutoDetectionFailed(f'{libname}: {funcname} could not be found')
    
    func.restype = ctypes.c_int

    if has_major_minor:
        func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        retval = func(major, minor)
        version = major.value * 1000 + minor.value * 10
    else:
        func.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version_ref = ctypes.c_int()
        retval = func(version_ref)
        version = version_ref.value

    if retval != 0:
        raise AutoDetectionFailed(f'{libname}: {funcname} returned error: {retval}')
    
    _log(f'Detected version: {version}')
    return version

def _get_cuda_version():
    """
    Tries to detect the CUDA version from various sources.
    First checks the CUDA_VERSION macro in the /usr/local/cuda/include/cuda.h file.
    if that fails, checks key Nvidia CUDA libraries such as libcudart, libcuda, etc.
    """
    # Try to get cuda version from cuda.h
    version = _get_version_from_cuda_header()
    if version is not None:
        return version
    
    # Try to get cuda version from installed CUDA runtime library
    libnames = ['libcudart.so.12', 'libcudart.so.11.0']
    try:
        version = _get_version_from_library(libnames, 'cudaRuntimeGetVersion')
        if version:
            return version
    except Exception as e:
        _log(f"Error: {e}")

    # Try to get cuda version from installed Nvidia libraries
    other_cuda_libs = {
        'libcublas.so': ['cublasGetVersion', ['libcublas.so.12', 'libcublas.so.11'], False],
        'libcusolver.so': ['cusolverGetVersion', ['libcusolver.so.12', 'libcusolver.so.11'], False],
        'libcusparse.so': ['cusparseGetVersion', ['libcusparse.so.12', 'libcusparse.so.11'], False],
        'libcurand.so': ['curandGetVersion', ['libcurand.so.12', 'libcurand.so.11'], False],
        'libcufft.so': ['cufftGetVersion', ['libcufft.so.12', 'libcufft.so.11'], False],
        'libnppc.so': ['nppiGetLibVersion', ['libnppc.so.12', 'libnppc.so.11'], False],
        'libnvrtc.so': ['nvrtcVersion', ['libnvrtc.so.12', 'libnvrtc.so.11.2', 'libnvrtc.so.11.1', 'libnvrtc.so.11.0'], True],
    }
    for libname, (funcname, libnames, has_major_minor) in other_cuda_libs.items():
        try:
            version = _get_version_from_library(libnames, funcname, has_major_minor)
            if version:
                return version
        except Exception as e:
            _log(f'Error detecting version from {libname}: {e}')

    _log('Autodetection failed')
    return None

def _find_installed_packages() -> List[str]:
    """Returns the list of out packages installed in the environment."""

    f = lambda x: ''.join([f"{PACKAGE_NAME}-cu", x])
    found = []

    for pkg in list(map(f, PACKAGE_SUPPORTED_CUDA_VER)):
        try:
            pkg_resources.get_distribution(pkg)
            found.append(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return found

def _cuda_version_to_package(ver):
    ver_str = int(str(ver)[:2])

    if ver_str < 11:
        raise AutoDetectionFailed(f'Your CUDA version ({ver}) is too old.')
    elif ver_str < 12:
        suffix = '11'
    elif ver_str < 13:
        suffix = '12'
    else:
        raise AutoDetectionFailed(f'Your CUDA version ({ver}) is too new.')
    
    return f'{PACKAGE_NAME}-cu{suffix}'

def infer_best_package(package_name: str,
                       package_supported_cuda_ver: List[str] = ['11', '12']) -> str:
    global PACKAGE_NAME, PACKAGE_SUPPORTED_CUDA_VER
    PACKAGE_NAME = package_name
    PACKAGE_SUPPORTED_CUDA_VER = sorted(package_supported_cuda_ver)

    # Find the existing wheel installation
    installed = _find_installed_packages()

    version = _get_cuda_version()
    if version is not None:
        to_install = _cuda_version_to_package(version)
    else:
        message = (
            "See below for the error message and instruction.\n\n\n" +
            "************************************************************************\n" +
            "ERROR: Unable to detect NVIDIA CUDA Toolkit installation\n" +
            f"ERROR: Explicitly specify CUDA version by: `pip install {PACKAGE_NAME}` run\n" +
            f"ERROR: `pip install {PACKAGE_NAME}-cuXX`, with XX being the major\n" +
            "ERROR: version of your CUDA Toolkit installation.\n" +
            "************************************************************************\n\n"
        )
        raise AutoDetectionFailed(message)
    
    # Disallow -cu11 & -cu12 wheels from coexisting
    if len(installed) > 1 or (len(installed) == 1 and installed[0] != to_install):
        raise AutoDetectionFailed(
            f'You already have the {PACKAGE_NAME} package(s) installed: \n'
            f'  {installed}\n'
            'while you attempt to install:\n'
            f'  {to_install}\n'
            'Please uninstall all of them first, then try reinstalling.')
    
    global PACKAGE_RESOLUTION, CUDA_RESOLUTION
    PACKAGE_RESOLUTION = to_install
    CUDA_RESOLUTION = version
    _log(f"Installing {to_install}...")
    return to_install


if _bdist_wheel is not None:

    # Technically we need a way to force reinstalling the sdist and ignored the cached wheel.
    # That said, I cannot reproduce the past-known behavior in my env. The sdist is always
    # reinstalled, though it could be due to certain combination of pip/setuptools/wheel/etc,
    # and it's still better to keep this WAR.

    class bdist_wheel(_bdist_wheel):

        # Adopted from https://discuss.python.org/t/wheel-caching-and-non-deterministic-builds/7687

        def finalize_options(self):
            super().finalize_options()

            # Use "cuXX" as the build tag to force re-running sdist if the
            # CUDA version in the user env has changed
            if PACKAGE_RESOLUTION is None:
                assert False, "something went wrong"
            build_tag = PACKAGE_RESOLUTION.split("-")[-1]

            # per PEP 427, build tag must start with a digit
            self.build_number = f"0_{build_tag}"
else:
    bdist_wheel = None
